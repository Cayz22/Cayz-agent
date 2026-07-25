"""
统一缓存层：基于 cachetools 的 TTL+LRU 缓存，线程安全。

为 LLM 调用、Embedding 向量化、RAG 检索提供短期结果复用，降低 API 成本与延迟。

设计要点：
- 每类缓存独立实例（LLM / Embedding / RAG），便于单独禁用或调整容量
- 内置命中/未命中计数，对接 monitor.py 暴露缓存命中率指标
- 显式 disable 开关：可通过配置 zero-cost 关闭某类缓存
- clear() 方法供知识库变更时主动失效 RAG 缓存
"""

import logging
import threading
from collections.abc import Callable
from typing import Any

from cachetools import TTLCache

from .config import get_settings
from .monitor import record_cache_hit, record_cache_miss

logger = logging.getLogger(__name__)


class MonitoredCache:
    """带命中率监控的 TTL+LRU 缓存包装器。

    使用 cachetools.TTLCache 同时支持：
    - TTL：超过 ttl 秒后条目自动失效
    - LRU：超过 maxsize 后淘汰最久未访问的条目

    所有操作通过 threading.Lock 保护，线程安全。
    """

    def __init__(self, name: str, maxsize: int, ttl: int, enabled: bool = True):
        """
        Args:
            name: 缓存名称（用于日志和指标标签）
            maxsize: 最大条目数
            ttl: 生存时间（秒）
            enabled: 是否启用；False 时所有 get 直接返回 MISS
        """
        self.name = name
        self.maxsize = maxsize
        self.ttl = ttl
        self.enabled = enabled
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()
        # P2-15：单 key 级细粒度锁，防止 cache stampede（雷群效应）
        # 多个并发线程对同一 key 未命中时，只允许一个执行 compute，其余等待
        self._key_locks: dict = {}
        self._key_locks_lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        """获取缓存值。命中时记录指标，未命中或禁用时返回 None。"""
        if not self.enabled:
            record_cache_miss(self.name)
            return None
        with self._lock:
            value = self._cache.get(key)
        if value is not None:
            record_cache_hit(self.name)
            logger.debug("缓存命中 [%s] key=%r", self.name, key)
            return value
        record_cache_miss(self.name)
        return None

    def set(self, key: Any, value: Any) -> None:
        """写入缓存。禁用时直接跳过。"""
        if not self.enabled:
            return
        with self._lock:
            self._cache[key] = value

    def _get_key_lock(self, key: Any) -> threading.Lock:
        """P2-15：获取单 key 级锁（防止 cache stampede）。

        使用外层 dict 锁保护 _key_locks 字典本身的线程安全。
        为避免 _key_locks 无限增长，条目过期后不会主动清理（影响极小，
        因 key 通常有限且为哈希字符串）。
        """
        with self._key_locks_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def get_or_compute(self, key: Any, compute: Callable[[], Any]) -> Any:
        """获取或计算模式：命中则直接返回，未命中则调用 compute 并写入缓存。

        P2-15 修复：使用单 key 级细粒度锁防止 cache stampede（雷群效应）。
        多个并发线程对同一 key 同时未命中时，只允许一个执行 compute()，
        其余线程等待后通过 double-check 命中缓存，避免重复计算。

        Args:
            key: 缓存键
            compute: 未命中时的计算函数（应无副作用或可重复执行）

        Returns:
            缓存值或 compute() 的结果
        """
        if not self.enabled:
            return compute()
        value = self.get(key)
        if value is not None:
            return value
        # P2-15：单 key 锁 + double-check，防止并发重复 compute
        with self._get_key_lock(key):
            value = self.get(key)  # double-check：等待期间可能已被其他线程填充
            if value is not None:
                return value
            value = compute()
            self.set(key, value)
            return value

    def clear(self) -> None:
        """清空所有缓存条目（用于知识库变更等场景主动失效）。"""
        with self._lock:
            cleared = len(self._cache)
            self._cache.clear()
        if cleared > 0:
            logger.info("缓存 [%s] 已清空 %d 个条目", self.name, cleared)

    def size(self) -> int:
        """返回当前缓存条目数"""
        with self._lock:
            return len(self._cache)

    def stats(self) -> dict:
        """返回缓存统计快照（不含命中率，命中率由 monitor 统计）"""
        with self._lock:
            return {
                "name": self.name,
                "enabled": self.enabled,
                "maxsize": self.maxsize,
                "ttl": self.ttl,
                "current_size": len(self._cache),
            }


# ============================================================
# 全局缓存实例（按类型隔离）
# ============================================================
_llm_cache: MonitoredCache | None = None
_embedding_cache: MonitoredCache | None = None
_rag_cache: MonitoredCache | None = None
_cache_lock = threading.Lock()


def get_llm_cache() -> MonitoredCache:
    """获取 LLM 调用缓存单例"""
    global _llm_cache
    if _llm_cache is None:
        with _cache_lock:
            if _llm_cache is None:
                s = get_settings()
                _llm_cache = MonitoredCache(
                    name="llm",
                    maxsize=s.cache_llm_maxsize,
                    ttl=s.cache_llm_ttl,
                    enabled=s.cache_llm_enabled,
                )
                logger.info(
                    "LLM 缓存已初始化: enabled=%s, maxsize=%d, ttl=%ds",
                    s.cache_llm_enabled,
                    s.cache_llm_maxsize,
                    s.cache_llm_ttl,
                )
    return _llm_cache


def get_embedding_cache() -> MonitoredCache:
    """获取 Embedding 向量化缓存单例"""
    global _embedding_cache
    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                s = get_settings()
                _embedding_cache = MonitoredCache(
                    name="embedding",
                    maxsize=s.cache_embedding_maxsize,
                    ttl=s.cache_embedding_ttl,
                    enabled=s.cache_embedding_enabled,
                )
                logger.info(
                    "Embedding 缓存已初始化: enabled=%s, maxsize=%d, ttl=%ds",
                    s.cache_embedding_enabled,
                    s.cache_embedding_maxsize,
                    s.cache_embedding_ttl,
                )
    return _embedding_cache


def get_rag_cache() -> MonitoredCache:
    """获取 RAG 检索缓存单例"""
    global _rag_cache
    if _rag_cache is None:
        with _cache_lock:
            if _rag_cache is None:
                s = get_settings()
                _rag_cache = MonitoredCache(
                    name="rag",
                    maxsize=s.cache_rag_maxsize,
                    ttl=s.cache_rag_ttl,
                    enabled=s.cache_rag_search_enabled,
                )
                logger.info(
                    "RAG 缓存已初始化: enabled=%s, maxsize=%d, ttl=%ds",
                    s.cache_rag_search_enabled,
                    s.cache_rag_maxsize,
                    s.cache_rag_ttl,
                )
    return _rag_cache


def invalidate_rag_cache() -> None:
    """失效 RAG 检索缓存（知识库上传/删除/更新后调用）"""
    get_rag_cache().clear()


def invalidate_all_caches() -> None:
    """失效所有缓存（管理端点/测试场景使用）"""
    get_llm_cache().clear()
    get_embedding_cache().clear()
    get_rag_cache().clear()


def reset_cache_singletons() -> None:
    """重置所有缓存单例（主要用于测试场景，让下个测试重新读取配置）"""
    global _llm_cache, _embedding_cache, _rag_cache
    with _cache_lock:
        _llm_cache = None
        _embedding_cache = None
        _rag_cache = None

"""
cache 模块单元测试：验证 TTL+LRU 缓存与命中率监控
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from cayz_agent.cache import (
    MonitoredCache,
    get_embedding_cache,
    get_llm_cache,
    get_rag_cache,
    invalidate_all_caches,
    invalidate_rag_cache,
    reset_cache_singletons,
)
from cayz_agent.monitor import get_registry, record_cache_hit, record_cache_miss


class TestMonitoredCache:
    """测试 MonitoredCache 数据结构"""

    def test_get_miss_returns_none(self):
        """未写入的 key 应返回 None"""
        c = MonitoredCache("test", maxsize=10, ttl=60)
        assert c.get("missing") is None

    def test_set_then_get_returns_value(self):
        """写入后应能读取"""
        c = MonitoredCache("test", maxsize=10, ttl=60)
        c.set("key1", "value1")
        assert c.get("key1") == "value1"

    def test_get_or_compute_caches_result(self):
        """get_or_compute 应缓存计算结果"""
        c = MonitoredCache("test", maxsize=10, ttl=60)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed"

        # 第一次：未命中，调用 compute
        result1 = c.get_or_compute("k", compute)
        assert result1 == "computed"
        assert call_count == 1

        # 第二次：命中缓存，不调用 compute
        result2 = c.get_or_compute("k", compute)
        assert result2 == "computed"
        assert call_count == 1

    def test_disabled_cache_always_misses(self):
        """禁用的缓存应总是返回 None 且不写入"""
        c = MonitoredCache("test", maxsize=10, ttl=60, enabled=False)
        c.set("key", "value")  # 应被跳过
        assert c.get("key") is None

    def test_disabled_get_or_compute_calls_compute(self):
        """禁用缓存时 get_or_compute 应每次调用 compute"""
        c = MonitoredCache("test", maxsize=10, ttl=60, enabled=False)
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return "computed"

        c.get_or_compute("k", compute)
        c.get_or_compute("k", compute)
        assert call_count == 2

    def test_clear_empties_cache(self):
        """clear 应清空所有条目"""
        c = MonitoredCache("test", maxsize=10, ttl=60)
        c.set("k1", "v1")
        c.set("k2", "v2")
        assert c.size() == 2
        c.clear()
        assert c.size() == 0
        assert c.get("k1") is None

    def test_lru_eviction_when_full(self):
        """超过 maxsize 后应淘汰最久未访问的条目"""
        c = MonitoredCache("test", maxsize=2, ttl=60)
        c.set("k1", "v1")
        c.set("k2", "v2")
        # 写入 k3 应淘汰 k1（最久未访问）
        c.set("k3", "v3")
        assert c.get("k1") is None
        assert c.get("k2") == "v2"
        assert c.get("k3") == "v3"

    def test_ttl_expiration(self):
        """TTL 过期后应返回 None"""
        c = MonitoredCache("test", maxsize=10, ttl=1)
        c.set("k", "v")
        assert c.get("k") == "v"
        # 等待 TTL 过期
        time.sleep(1.1)
        assert c.get("k") is None

    def test_stats_returns_metadata(self):
        """stats 应返回缓存元数据"""
        c = MonitoredCache("test", maxsize=10, ttl=60, enabled=True)
        c.set("k", "v")
        stats = c.stats()
        assert stats["name"] == "test"
        assert stats["enabled"] is True
        assert stats["maxsize"] == 10
        assert stats["ttl"] == 60
        assert stats["current_size"] == 1


class TestCacheMetrics:
    """测试缓存命中率监控"""

    def setup_method(self):
        """每个测试前重置监控注册表"""
        get_registry().reset()

    def test_hit_increments_hit_counter(self):
        """命中时应递增 cache_hits 指标"""
        c = MonitoredCache("llm", maxsize=10, ttl=60)
        c.set("k", "v")
        c.get("k")  # 命中
        reg = get_registry()
        assert reg.cache_hits["llm"].get() == 1
        assert reg.cache_misses["llm"].get() == 0

    def test_miss_increments_miss_counter(self):
        """未命中时应递增 cache_misses 指标"""
        c = MonitoredCache("llm", maxsize=10, ttl=60)
        c.get("missing")  # 未命中
        reg = get_registry()
        assert reg.cache_hits["llm"].get() == 0
        assert reg.cache_misses["llm"].get() == 1

    def test_disabled_cache_records_miss(self):
        """禁用的缓存 get 时应记录为 miss"""
        c = MonitoredCache("llm", maxsize=10, ttl=60, enabled=False)
        c.get("any")
        reg = get_registry()
        assert reg.cache_misses["llm"].get() == 1

    def test_multiple_caches_tracked_separately(self):
        """不同名称的缓存应分别统计"""
        c1 = MonitoredCache("llm", maxsize=10, ttl=60)
        c2 = MonitoredCache("rag", maxsize=10, ttl=60)
        c1.set("k", "v")
        c2.set("k", "v")
        c1.get("k")  # llm 命中
        c2.get("missing")  # rag 未命中

        reg = get_registry()
        assert reg.cache_hits["llm"].get() == 1
        assert reg.cache_misses["rag"].get() == 1


class TestGlobalCacheInstances:
    """测试全局缓存单例"""

    def test_get_llm_cache_returns_singleton(self):
        """get_llm_cache 应返回单例"""
        reset_cache_singletons()
        c1 = get_llm_cache()
        c2 = get_llm_cache()
        assert c1 is c2
        assert c1.name == "llm"

    def test_get_embedding_cache_returns_singleton(self):
        """get_embedding_cache 应返回单例"""
        reset_cache_singletons()
        c1 = get_embedding_cache()
        c2 = get_embedding_cache()
        assert c1 is c2
        assert c1.name == "embedding"

    def test_get_rag_cache_returns_singleton(self):
        """get_rag_cache 应返回单例"""
        reset_cache_singletons()
        c1 = get_rag_cache()
        c2 = get_rag_cache()
        assert c1 is c2
        assert c1.name == "rag"

    def test_reset_cache_singletons_creates_new_instances(self):
        """reset_cache_singletons 后应创建新实例"""
        reset_cache_singletons()
        c1 = get_llm_cache()
        reset_cache_singletons()
        c2 = get_llm_cache()
        assert c1 is not c2

    def test_invalidate_rag_cache_only_clears_rag(self):
        """invalidate_rag_cache 应只清空 RAG 缓存"""
        reset_cache_singletons()
        llm = get_llm_cache()
        rag = get_rag_cache()
        llm.set("k", "v")
        rag.set("k", "v")
        invalidate_rag_cache()
        assert llm.get("k") == "v"  # LLM 缓存不受影响
        assert rag.get("k") is None  # RAG 缓存已清空

    def test_invalidate_all_caches_clears_everything(self):
        """invalidate_all_caches 应清空所有缓存"""
        reset_cache_singletons()
        llm = get_llm_cache()
        emb = get_embedding_cache()
        rag = get_rag_cache()
        llm.set("k", "v")
        emb.set("k", "v")
        rag.set("k", "v")
        invalidate_all_caches()
        assert llm.get("k") is None
        assert emb.get("k") is None
        assert rag.get("k") is None


class TestCacheConfigIntegration:
    """测试缓存与配置的集成"""

    def test_cache_uses_config_values(self):
        """缓存应使用 Settings 中的 maxsize 和 ttl"""
        reset_cache_singletons()
        from cayz_agent.config import get_settings

        s = get_settings()
        llm = get_llm_cache()
        assert llm.maxsize == s.cache_llm_maxsize
        assert llm.ttl == s.cache_llm_ttl
        assert llm.enabled == s.cache_llm_enabled

    def test_cache_respects_disabled_flag(self):
        """cache_*_enabled=False 时缓存应禁用"""
        reset_cache_singletons()
        with patch("cayz_agent.config.Settings") as mock_settings_cls:
            mock_s = MagicMock()
            mock_s.cache_llm_enabled = False
            mock_s.cache_llm_maxsize = 10
            mock_s.cache_llm_ttl = 60
            mock_settings_cls.return_value = mock_s

            # 重新获取单例（会读取 mock 配置）
            cache = get_llm_cache()
            assert cache.enabled is False


class TestCacheMetricsInPrometheusExport:
    """测试缓存指标出现在 Prometheus 导出中"""

    def setup_method(self):
        get_registry().reset()

    def test_export_contains_cache_metrics(self):
        """Prometheus 导出应包含缓存命中/未命中指标"""
        from cayz_agent.monitor import export_prometheus

        # 触发一些缓存操作
        record_cache_hit("llm")
        record_cache_miss("llm")

        output = export_prometheus()
        assert "cayz_cache_hits_total" in output
        assert "cayz_cache_misses_total" in output
        assert 'cache="llm"' in output

    def test_metrics_summary_includes_cache_stats(self):
        """metrics 摘要应包含缓存命中率"""
        from cayz_agent.monitor import get_metrics_summary

        record_cache_hit("llm")
        record_cache_hit("llm")
        record_cache_miss("llm")

        summary = get_metrics_summary()
        assert "cache" in summary
        assert "llm" in summary["cache"]
        assert summary["cache"]["llm"]["hits"] == 2
        assert summary["cache"]["llm"]["misses"] == 1
        # 命中率 = 2 / (2 + 1) ≈ 0.6667
        assert summary["cache"]["llm"]["hit_rate"] == 0.6667

    def test_metrics_summary_empty_cache_when_no_activity(self):
        """无缓存活动时 cache 字段应为空 dict"""
        from cayz_agent.monitor import get_metrics_summary

        summary = get_metrics_summary()
        assert summary["cache"] == {}

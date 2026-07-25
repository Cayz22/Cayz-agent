"""
监控指标收集模块

- 线程安全的内存指标存储
- 支持 Prometheus 文本格式导出
- 覆盖：请求计数、延迟、Token 用量、工具调用、错误率、活跃会话

不引入外部依赖，纯 Python 实现。
"""
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Counter:
    """单调递增计数器"""
    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: int = 1):
        with self._lock:
            self.value += amount

    def get(self) -> int:
        with self._lock:
            return self.value


@dataclass
class Histogram:
    """延迟分布统计（简化版：记录 count / sum / buckets）

    P0 修复：_bucket_counts 按桶下标存储「累计计数」（即 value <= b 的样本数），
    这与 Prometheus histogram 的语义一致；导出时直接输出，不再二次累加。
    """
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    count: int = 0
    total: float = 0.0
    _buckets: list = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0], repr=False)
    # P0：按桶下标存储累计计数；初始化为 0 保证未观测的桶也输出（Prometheus 要求所有 le 桶存在）
    _bucket_counts: dict = field(default_factory=dict, repr=False)

    def observe(self, value: float):
        with self._lock:
            self.count += 1
            self.total += value
            # P0：每个 bucket 是「value <= b」的累计计数（Prometheus 语义）
            # 遍历所有 bucket，满足条件的 +1，导出时直接输出
            for b in self._buckets:
                if value <= b:
                    key = f"le_{b}"
                    self._bucket_counts[key] = self._bucket_counts.get(key, 0) + 1

    def get_stats(self) -> dict:
        with self._lock:
            avg = self.total / self.count if self.count > 0 else 0.0
            # P0：补齐所有桶为 0，避免导出缺失 le 桶（Prometheus 要求桶序列完整）
            buckets = {}
            for b in self._buckets:
                key = f"le_{b}"
                buckets[key] = self._bucket_counts.get(key, 0)
            return {
                "count": self.count,
                "sum": self.total,
                "avg": avg,
                "buckets": buckets,
            }


@dataclass
class Gauge:
    """可增可减的指标"""
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, value: float):
        with self._lock:
            self.value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self.value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self.value -= amount

    def get(self) -> float:
        with self._lock:
            return self.value


# ============================================================
# 全局指标注册表
# ============================================================

class MetricsRegistry:
    """全局指标注册表，单例模式

    P1 修复：将所有初始化逻辑移入 __new__（在锁内完成），避免 __init__ 与 __new__
    分离导致的竞态——旧实现中 __init__ 的 _initialized 检查不在锁内，两个线程同时
    首次调用时 Thread B 可能重建所有 Counter 致指标归零。
    """

    _instance: Optional["MetricsRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                # P1：在锁内完成所有初始化，避免 __init__ 竞态
                instance._init_metrics()
                cls._instance = instance
            return cls._instance

    def __init__(self):
        # P1：初始化已在 __new__ 中完成，__init__ 幂等无操作
        # （Python 会在每次 MyClass() 调用时自动调用 __init__，无法阻止）
        pass

    def _init_metrics(self):
        """初始化所有指标（仅在 __new__ 首次创建实例时调用一次，在锁内执行）"""
        # 请求指标
        self.requests_total = Counter()
        self.requests_by_type: dict[str, Counter] = defaultdict(Counter)
        self.request_errors = Counter()
        self.request_latency = Histogram()

        # Token 用量
        self.tokens_input = Counter()
        self.tokens_output = Counter()
        self.tokens_total = Counter()

        # 工具调用
        self.tool_calls_total = Counter()
        self.tool_calls_by_name: dict[str, Counter] = defaultdict(Counter)
        self.tool_call_errors = Counter()
        self.tool_call_latency = Histogram()

        # 路由指标（多 Agent）
        self.route_counts: dict[str, Counter] = defaultdict(Counter)

        # 会话指标
        self.active_sessions = Gauge()
        self.sessions_deleted = Counter()
        # P3：会话时长分布 histogram（秒），用于分析会话质量
        # 桶：1s / 5s / 10s / 30s / 60s / 300s / 1800s / 3600s / +Inf
        self.session_duration = Histogram()
        self.session_duration._buckets = [1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 1800.0, 3600.0]

        # 验证指标
        self.validation_failures = Counter()
        self.input_rejected = Counter()

        # 重试指标
        self.retry_attempts = Counter()

        # 知识库指标
        self.knowledge_uploads = Counter()
        self.knowledge_deletes = Counter()
        self.knowledge_chunks_total = Counter()
        # P3：RAG 检索指标（独立于 cache_hits/misses，衡量检索总量与延迟）
        self.rag_searches_total = Counter()
        self.rag_search_errors = Counter()
        self.rag_search_latency = Histogram()
        # P3：RAG 检索结果数分布（衡量检索质量，过少可能召回不足）
        self.rag_search_results = Histogram()
        self.rag_search_results._buckets = [0, 1, 3, 5, 10, 20, 50]

        # 缓存指标（按缓存名称分桶：llm / embedding / rag）
        self.cache_hits: dict[str, Counter] = defaultdict(Counter)
        self.cache_misses: dict[str, Counter] = defaultdict(Counter)

        # 启动时间
        self.start_time = time.time()

    def reset(self):
        """重置所有指标（用于测试）"""
        with self._lock:
            self._init_metrics()


def get_registry() -> MetricsRegistry:
    """获取全局指标注册表"""
    return MetricsRegistry()


# ============================================================
# 便捷记录函数
# ============================================================

def record_request(request_type: str = "chat", success: bool = True, latency: float = 0.0):
    """记录一次 API/对话请求"""
    reg = get_registry()
    reg.requests_total.inc()
    reg.requests_by_type[request_type].inc()
    reg.request_latency.observe(latency)
    if not success:
        reg.request_errors.inc()


def record_token_usage(input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = 0):
    """记录一次 LLM 调用的 Token 用量"""
    reg = get_registry()
    reg.tokens_input.inc(input_tokens)
    reg.tokens_output.inc(output_tokens)
    reg.tokens_total.inc(total_tokens)


def record_tool_call(tool_name: str, success: bool = True, latency: float = 0.0):
    """记录一次工具调用"""
    reg = get_registry()
    reg.tool_calls_total.inc()
    reg.tool_calls_by_name[tool_name].inc()
    reg.tool_call_latency.observe(latency)
    if not success:
        reg.tool_call_errors.inc()


def record_route(route: str):
    """记录一次路由决策"""
    reg = get_registry()
    reg.route_counts[route].inc()


def record_validation_failure():
    """记录一次输入验证失败"""
    reg = get_registry()
    reg.validation_failures.inc()
    reg.input_rejected.inc()


def record_retry():
    """记录一次重试"""
    reg = get_registry()
    reg.retry_attempts.inc()


def record_session_start():
    """记录会话开始（活跃会话数 +1）"""
    reg = get_registry()
    reg.active_sessions.inc()


def record_session_end(duration: float = 0.0):
    """记录会话结束（活跃会话数 -1）

    P3：可选记录会话时长（秒），用于分析会话质量分布。
    时长定义：从 record_session_start 到 record_session_end 的墙钟时间差。
    由调用方计算并传入（避免本模块维护 per-session 起始时间戳的复杂性）。

    Args:
        duration: 会话时长（秒），<=0 表示不记录 histogram
    """
    reg = get_registry()
    reg.active_sessions.dec()
    if duration > 0:
        reg.session_duration.observe(duration)


def record_session_deleted():
    """记录会话被删除"""
    reg = get_registry()
    reg.sessions_deleted.inc()


def record_knowledge_upload(chunks: int = 0):
    """记录知识库文档上传"""
    reg = get_registry()
    reg.knowledge_uploads.inc()
    if chunks > 0:
        reg.knowledge_chunks_total.inc(chunks)


def record_knowledge_delete(chunks: int = 0):
    """记录知识库文档删除"""
    reg = get_registry()
    reg.knowledge_deletes.inc()
    if chunks > 0:
        reg.knowledge_chunks_total.inc(-chunks)


def record_cache_hit(cache_name: str):
    """记录一次缓存命中"""
    reg = get_registry()
    reg.cache_hits[cache_name].inc()


def record_cache_miss(cache_name: str):
    """记录一次缓存未命中"""
    reg = get_registry()
    reg.cache_misses[cache_name].inc()


# ============================================================
# P3 RAG 检索指标
# ============================================================

def record_rag_search(success: bool = True, latency: float = 0.0, result_count: int = 0):
    """记录一次 RAG 检索

    Args:
        success: 检索是否成功（False 表示底层异常，计入 rag_search_errors）
        latency: 检索延迟（秒）
        result_count: 返回的文档数（用于分析召回质量）
    """
    reg = get_registry()
    reg.rag_searches_total.inc()
    if latency > 0:
        reg.rag_search_latency.observe(latency)
    if not success:
        reg.rag_search_errors.inc()
    # result_count 作为 histogram 观测值（即使 success=False 也可记录 0）
    reg.rag_search_results.observe(result_count)


# ============================================================
# Prometheus 格式导出
# ============================================================

def _format_counter(name: str, help_text: str, value: int, labels: str = "") -> str:
    lines = [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} counter",
    ]
    if labels:
        lines.append(f'{name}{{{labels}}} {value}')
    else:
        lines.append(f"{name} {value}")
    return "\n".join(lines)


def _format_histogram(name: str, help_text: str, stats: dict, labels: str = "") -> str:
    lines = [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} histogram",
    ]
    # P0 修复：stats["buckets"] 已是累计计数（value <= b 的样本数），直接输出，
    # 不再二次累加（旧代码 cumulative += ... 会导致指标严重夸大）
    # 按 bucket 上界数值升序输出
    def _bucket_value(key: str) -> float:
        return float(key.replace("le_", ""))

    for bucket_bound in sorted(stats["buckets"].keys(), key=_bucket_value):
        le = bucket_bound.replace("le_", "")
        count = stats["buckets"][bucket_bound]
        if labels:
            lines.append(f'{name}_bucket{{{labels},le="{le}"}} {count}')
        else:
            lines.append(f'{name}_bucket{{le="{le}"}} {count}')
    # +Inf bucket
    if labels:
        lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {stats["count"]}')
    else:
        lines.append(f'{name}_bucket{{le="+Inf"}} {stats["count"]}')
    lines.append(f"{name}_sum {stats['sum']:.6f}")
    lines.append(f"{name}_count {stats['count']}")
    return "\n".join(lines)


def _format_gauge(name: str, help_text: str, value: float, labels: str = "") -> str:
    lines = [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
    ]
    if labels:
        lines.append(f'{name}{{{labels}}} {value}')
    else:
        lines.append(f"{name} {value}")
    return "\n".join(lines)


def export_prometheus() -> str:
    """导出所有指标为 Prometheus 文本格式

    P2-16 修复：迭代 defaultdict 时使用 list() 快照 keys，
    防止并发请求通过 defaultdict.__missing__ 新增 key 触发
    "dictionary changed size during iteration" RuntimeError。
    """
    reg = get_registry()
    parts = []

    # 请求指标
    parts.append(_format_counter(
        "cayz_requests_total", "Total number of requests", reg.requests_total.get()
    ))
    # P2-16：list() 快照 keys，防止并发插入导致迭代异常
    for req_type in list(reg.requests_by_type.keys()):
        counter = reg.requests_by_type[req_type]
        parts.append(_format_counter(
            "cayz_requests_total", "Total number of requests",
            counter.get(), f'type="{req_type}"'
        ))
    parts.append(_format_counter(
        "cayz_request_errors_total", "Total number of failed requests",
        reg.request_errors.get()
    ))
    parts.append(_format_histogram(
        "cayz_request_latency_seconds", "Request latency distribution",
        reg.request_latency.get_stats()
    ))

    # Token 用量
    parts.append(_format_counter(
        "cayz_tokens_input_total", "Total input tokens consumed",
        reg.tokens_input.get()
    ))
    parts.append(_format_counter(
        "cayz_tokens_output_total", "Total output tokens consumed",
        reg.tokens_output.get()
    ))
    parts.append(_format_counter(
        "cayz_tokens_total", "Total tokens consumed",
        reg.tokens_total.get()
    ))

    # 工具调用
    parts.append(_format_counter(
        "cayz_tool_calls_total", "Total tool calls",
        reg.tool_calls_total.get()
    ))
    # P2-16：list() 快照
    for tool_name in list(reg.tool_calls_by_name.keys()):
        counter = reg.tool_calls_by_name[tool_name]
        parts.append(_format_counter(
            "cayz_tool_calls_total", "Total tool calls",
            counter.get(), f'tool="{tool_name}"'
        ))
    parts.append(_format_counter(
        "cayz_tool_call_errors_total", "Total tool call errors",
        reg.tool_call_errors.get()
    ))
    parts.append(_format_histogram(
        "cayz_tool_call_latency_seconds", "Tool call latency distribution",
        reg.tool_call_latency.get_stats()
    ))

    # 路由指标
    # P2-16：list() 快照
    for route in list(reg.route_counts.keys()):
        counter = reg.route_counts[route]
        parts.append(_format_counter(
            "cayz_route_total", "Total route decisions",
            counter.get(), f'route="{route}"'
        ))

    # 会话
    parts.append(_format_gauge(
        "cayz_active_sessions", "Number of active sessions",
        reg.active_sessions.get()
    ))
    parts.append(_format_counter(
        "cayz_sessions_deleted_total", "Total sessions deleted",
        reg.sessions_deleted.get()
    ))
    # P3：会话时长分布
    parts.append(_format_histogram(
        "cayz_session_duration_seconds", "Session duration distribution in seconds",
        reg.session_duration.get_stats()
    ))

    # 验证
    parts.append(_format_counter(
        "cayz_validation_failures_total", "Total input validation failures",
        reg.validation_failures.get()
    ))

    # 重试
    parts.append(_format_counter(
        "cayz_retry_attempts_total", "Total retry attempts",
        reg.retry_attempts.get()
    ))

    # 知识库
    parts.append(_format_counter(
        "cayz_knowledge_uploads_total", "Total knowledge document uploads",
        reg.knowledge_uploads.get()
    ))
    parts.append(_format_counter(
        "cayz_knowledge_deletes_total", "Total knowledge document deletes",
        reg.knowledge_deletes.get()
    ))
    parts.append(_format_counter(
        "cayz_knowledge_chunks_total", "Total knowledge chunks in vector store",
        reg.knowledge_chunks_total.get()
    ))
    # P3：RAG 检索指标
    parts.append(_format_counter(
        "cayz_rag_searches_total", "Total RAG searches performed",
        reg.rag_searches_total.get()
    ))
    parts.append(_format_counter(
        "cayz_rag_search_errors_total", "Total RAG search errors",
        reg.rag_search_errors.get()
    ))
    parts.append(_format_histogram(
        "cayz_rag_search_latency_seconds", "RAG search latency distribution",
        reg.rag_search_latency.get_stats()
    ))
    parts.append(_format_histogram(
        "cayz_rag_search_results", "RAG search result count distribution",
        reg.rag_search_results.get_stats()
    ))

    # 缓存指标
    # P2-16：list() 快照
    for cache_name in list(reg.cache_hits.keys()):
        counter = reg.cache_hits[cache_name]
        parts.append(_format_counter(
            "cayz_cache_hits_total", "Total cache hits",
            counter.get(), f'cache="{cache_name}"'
        ))
    for cache_name in list(reg.cache_misses.keys()):
        counter = reg.cache_misses[cache_name]
        parts.append(_format_counter(
            "cayz_cache_misses_total", "Total cache misses",
            counter.get(), f'cache="{cache_name}"'
        ))

    # 运行时长
    uptime = time.time() - reg.start_time
    parts.append(_format_gauge(
        "cayz_uptime_seconds", "Service uptime in seconds",
        round(uptime, 2)
    ))

    return "\n\n".join(parts) + "\n"


def get_metrics_summary() -> dict:
    """获取指标摘要（用于 /health 或日志输出）

    P2-16 修复：使用 list() 快照 keys 防止并发迭代异常。
    """
    reg = get_registry()
    latency_stats = reg.request_latency.get_stats()

    # 计算各类缓存的命中率
    # P2-16：list() 快照 keys
    cache_stats: dict[str, dict] = {}
    all_cache_names = list(set(reg.cache_hits.keys()) | set(reg.cache_misses.keys()))
    for name in all_cache_names:
        hits = reg.cache_hits.get(name, Counter()).get()
        misses = reg.cache_misses.get(name, Counter()).get()
        total = hits + misses
        rate = round(hits / total, 4) if total > 0 else 0.0
        cache_stats[name] = {
            "hits": hits,
            "misses": misses,
            "hit_rate": rate,
        }

    return {
        "requests_total": reg.requests_total.get(),
        "request_errors": reg.request_errors.get(),
        "avg_latency_seconds": round(latency_stats["avg"], 4),
        "tokens_total": reg.tokens_total.get(),
        "tool_calls_total": reg.tool_calls_total.get(),
        "tool_call_errors": reg.tool_call_errors.get(),
        "validation_failures": reg.validation_failures.get(),
        "retry_attempts": reg.retry_attempts.get(),
        "active_sessions": reg.active_sessions.get(),
        "sessions_deleted": reg.sessions_deleted.get(),
        "knowledge_uploads": reg.knowledge_uploads.get(),
        "knowledge_deletes": reg.knowledge_deletes.get(),
        "knowledge_chunks": reg.knowledge_chunks_total.get(),
        # P3：RAG 检索摘要
        "rag_searches": reg.rag_searches_total.get(),
        "rag_search_errors": reg.rag_search_errors.get(),
        "cache": cache_stats,
        "uptime_seconds": round(time.time() - reg.start_time, 2),
    }

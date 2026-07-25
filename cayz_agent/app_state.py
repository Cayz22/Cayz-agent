"""
P3 应用状态管理：就绪标志 + 资源清理钩子。

- _ready 标志：启动时为 False，预热完成后置 True；shutdown 时立即置 False
- _cleanup_hooks：注册资源清理函数（后进先执行），shutdown 时统一调用
- _app_started_at：应用启动时间戳，用于 /health/ready 报告运行时长

设计要点：
- 不引入全局可变状态污染：仅通过函数接口暴露
- 清理钩子 LIFO 执行：后注册的先清理（依赖关系：后注册的通常依赖先注册的资源）
- 清理函数失败不阻塞其他清理：每个 hook 独立 try/except
"""

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# 应用就绪标志：False 时 /health/ready 返回 503
# 启动时为 False，lifespan startup 完成后置 True；shutdown 时立即置 False
_ready: bool = False
_app_started_at: float = time.time()
# 资源清理钩子列表（LIFO 执行）
_cleanup_hooks: list[Callable[[], None]] = []
_lock = threading.Lock()


def set_ready(value: bool) -> None:
    """设置应用就绪标志"""
    global _ready
    with _lock:
        _ready = value
        if value:
            logger.info("应用已就绪，开始接收流量")
        else:
            logger.info("应用标记为未就绪，停止接收新流量")


def is_ready() -> bool:
    """返回应用是否就绪"""
    with _lock:
        return _ready


def get_app_started_at() -> float:
    """返回应用启动时间戳"""
    return _app_started_at


def register_cleanup(hook: Callable[[], None]) -> None:
    """注册资源清理函数（shutdown 时 LIFO 调用）

    使用场景：
    - SessionManager.close()
    - LangGraph SqliteSaver 连接关闭
    - ChromaDB client 关闭
    - 其他需要显式释放的资源
    """
    with _lock:
        _cleanup_hooks.append(hook)


def run_cleanups(timeout: float = 30.0) -> int:
    """执行所有注册的清理钩子（LIFO 顺序）。

    Args:
        timeout: 总超时时间（秒），超时后未执行的钩子跳过

    Returns:
        成功执行的钩子数
    """
    with _lock:
        hooks = list(reversed(_cleanup_hooks))  # LIFO：后注册的先执行
        _cleanup_hooks.clear()

    executed = 0
    deadline = time.time() + timeout
    for hook in hooks:
        remaining = deadline - time.time()
        if remaining <= 0:
            logger.warning("清理超时，剩余 %d 个钩子未执行", len(hooks) - executed)
            break
        try:
            hook()
            executed += 1
        except Exception:
            logger.exception("清理钩子执行失败: %s", getattr(hook, "__name__", hook))
    logger.info("资源清理完成（%d/%d 个钩子成功执行）", executed, len(hooks))
    return executed


def reset() -> None:
    """重置所有状态（仅用于测试）"""
    global _ready, _app_started_at, _cleanup_hooks
    with _lock:
        _ready = False
        _app_started_at = time.time()
        _cleanup_hooks = []

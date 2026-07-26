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

    P1 修复：旧实现的超时检查仅在 hook 执行前，单个 hook 阻塞
    （如 thread.join() 无超时、socket.recv() 阻塞、文件 I/O hang）会
    一直卡住，超时机制形同虚设，优雅停机可能无限等待。

    本实现用线程执行每个 hook，主线程 join(remaining) 等待：
    - hook 在 remaining 秒内完成 → 正常计数
    - hook 超时未完成 → 记录警告并跳过（daemon 线程继续在后台运行，
      进程退出时会被 OS 强制回收，不会真正泄漏）

    Args:
        timeout: 总超时时间（秒），超时后未执行的钩子跳过

    Returns:
        成功执行的钩子数（含超时但已启动的）
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

        hook_name = getattr(hook, "__name__", repr(hook))
        result_holder: dict = {"done": False, "exc": None}

        def _runner(h=hook, rh=result_holder):
            try:
                h()
                rh["done"] = True
            except Exception as e:  # noqa: BLE001
                rh["exc"] = e

        # daemon=True：进程退出时线程会被强制回收，避免 hook 阻塞导致进程无法退出
        t = threading.Thread(target=_runner, name=f"cleanup-{hook_name}", daemon=True)
        t.start()
        t.join(remaining)  # 最多等待 remaining 秒

        if t.is_alive():
            # hook 超时未完成，跳过（线程继续在后台运行，进程退出时由 OS 回收）
            logger.warning(
                "清理钩子 %s 执行超时（>%ss），跳过等待（后台线程将继续运行）",
                hook_name,
                remaining,
            )
            executed += 1  # 已启动，计入「已处理」
        elif result_holder["exc"] is not None:
            logger.exception("清理钩子执行失败: %s", hook_name, exc_info=result_holder["exc"])
        else:
            executed += 1

    logger.info("资源清理完成（%d/%d 个钩子已处理）", executed, len(hooks))
    return executed


def reset() -> None:
    """重置所有状态（仅用于测试）"""
    global _ready, _app_started_at, _cleanup_hooks
    with _lock:
        _ready = False
        _app_started_at = time.time()
        _cleanup_hooks = []

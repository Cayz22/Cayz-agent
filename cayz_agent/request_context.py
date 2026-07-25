"""
P3 请求上下文管理：基于 contextvars 的 request_id 注入。

- request_id 通过 contextvars 在异步调用链中传递，线程安全
- LoggerAdapter / LogFilter 自动将 request_id 写入日志记录
- 中间件设置 request_id，整个请求生命周期内的日志都带该字段

设计要点：
- contextvars 在 asyncio.Task 创建时自动 copy context，无需手动传递
- 中间件 set_request_id() 后，下游 await 调用自动继承
- 异常处理器、后台任务可通过 get_request_id() 读取（无则返回 None）
"""

import contextvars
import logging
import uuid

# 全局 contextvar：每个请求独立持有 request_id
_request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


def set_request_id(request_id: str | None) -> None:
    """设置当前上下文的 request_id"""
    _request_id_var.set(request_id)


def get_request_id() -> str | None:
    """获取当前上下文的 request_id（无则返回 None）"""
    return _request_id_var.get()


def new_request_id() -> str:
    """生成新的 request_id（UUID4 去横线，32 字符）"""
    return uuid.uuid4().hex


class RequestIdLogFilter(logging.Filter):
    """日志过滤器：将当前上下文的 request_id 注入到 LogRecord。

    与 SanitizingLogFilter 协同工作（均挂在 root logger 上）：
    - SanitizingLogFilter 负责脱敏
    - 本 Filter 负责 enrich（注入 request_id 字段）

    注入后：
    - text 格式：可通过 %(request_id)s 引用（默认格式不含此字段，需自定义 format）
    - json 格式：_JsonFormatter 会自动合并 extra 字段输出 request_id
    """

    def filter(self, record: logging.LogRecord) -> bool:
        rid = _request_id_var.get()
        # 仅在未显式设置时注入，允许调用方通过 extra={"request_id": ...} 覆盖
        if rid and not hasattr(record, "request_id"):
            record.request_id = rid
        return True

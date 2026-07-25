"""
自定义异常体系

层次结构：
    CayzAgentError                # 所有自定义异常的基类
    ├── ConfigError               # 配置错误
    ├── LLMError                  # LLM 调用相关
    │   └── LLMRateLimitError     # 限流（可重试）
    ├── ToolError                 # 工具执行相关
    ├── RAGError                  # 知识库相关
    │   ├── RAGConnectionError    # 向量库连接失败
    │   └── RAGIngestError        # 文档入库失败
    └── IntegrationError          # 业务系统集成相关
        ├── CRMError              # CRM 调用失败
        ├── NotifyError           # 企业微信通知失败
        └── EmailError            # 邮件发送失败

每个异常支持 message + cause + 可重试标记，便于上层统一处理。
"""


class CayzAgentError(Exception):
    """所有 cayz-agent 自定义异常的基类"""

    def __init__(self, message: str = "", *, cause: Exception | None = None, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.retryable = retryable

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (cause: {self.cause})"
        return self.message


# ===== 配置 =====

class ConfigError(CayzAgentError):
    """配置错误（缺失/无效/格式错误）"""


# ===== LLM =====

class LLMError(CayzAgentError):
    """LLM 调用相关错误"""


class LLMRateLimitError(LLMError):
    """LLM 限流错误（通常可重试）"""

    def __init__(self, message: str = "LLM 请求被限流", **kwargs):
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


# ===== 工具 =====

class ToolError(CayzAgentError):
    """工具执行错误"""


# ===== RAG =====

class RAGError(CayzAgentError):
    """RAG 知识库相关错误"""


class RAGConnectionError(RAGError):
    """向量数据库连接失败"""

    def __init__(self, message: str = "ChromaDB 连接失败", **kwargs):
        kwargs.setdefault("retryable", True)
        super().__init__(message, **kwargs)


class RAGIngestError(RAGError):
    """文档入库失败"""


# ===== 业务系统集成 =====

class IntegrationError(CayzAgentError):
    """业务系统集成基类"""


class CRMError(IntegrationError):
    """CRM 系统调用失败"""


class NotifyError(IntegrationError):
    """企业微信通知发送失败"""


class EmailError(IntegrationError):
    """邮件发送失败"""

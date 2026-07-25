"""
cayz-agent：具备持久化记忆、联网搜索、知识库检索（RAG）、多模型支持与多 Agent 协作的企业级 Agent
"""
from .config import get_settings, setup_logging, Settings
from .graph import create_graph, AgentState
from .multi_agent import create_multi_agent_graph, MultiAgentState
from .llm import create_llm, list_supported_providers
from .tools import AGENT_TOOLS
from .sanitizers import sanitize_text, sanitize_exception, contains_harmful_content
from .validators import validate_user_input, validate_search_query, InputValidationError
from .retry import retry_on_error, log_execution
from .monitor import (
    get_registry,
    record_request,
    record_token_usage,
    record_tool_call,
    record_route,
    record_validation_failure,
    record_retry,
    export_prometheus,
    get_metrics_summary,
)
from .alerts import get_alert_manager, check_alerts, Alert, AlertLevel
from .integrations import (
    CRMClient,
    get_crm_client,
    WeChatNotifier,
    get_notifier,
    EmailSender,
    get_email_sender,
)
from .middleware import APIKeyAuthMiddleware, RateLimitMiddleware, setup_middleware
from .session import SessionManager, SessionInfo, get_session_manager
from .exceptions import (
    CayzAgentError,
    ConfigError,
    LLMError,
    LLMRateLimitError,
    ToolError,
    RAGError,
    RAGConnectionError,
    RAGIngestError,
    IntegrationError,
    CRMError,
    NotifyError,
    EmailError,
)

__version__ = "1.0.0"

__all__ = [
    "get_settings",
    "setup_logging",
    "Settings",
    "create_graph",
    "AgentState",
    "create_multi_agent_graph",
    "MultiAgentState",
    "create_llm",
    "list_supported_providers",
    "AGENT_TOOLS",
    "sanitize_text",
    "sanitize_exception",
    "contains_harmful_content",
    "validate_user_input",
    "validate_search_query",
    "InputValidationError",
    "retry_on_error",
    "log_execution",
    "get_registry",
    "record_request",
    "record_token_usage",
    "record_tool_call",
    "record_route",
    "record_validation_failure",
    "record_retry",
    "export_prometheus",
    "get_metrics_summary",
    "get_alert_manager",
    "check_alerts",
    "Alert",
    "AlertLevel",
    "CRMClient",
    "get_crm_client",
    "WeChatNotifier",
    "get_notifier",
    "EmailSender",
    "get_email_sender",
    "APIKeyAuthMiddleware",
    "RateLimitMiddleware",
    "setup_middleware",
    "CayzAgentError",
    "ConfigError",
    "LLMError",
    "LLMRateLimitError",
    "ToolError",
    "RAGError",
    "RAGConnectionError",
    "RAGIngestError",
    "IntegrationError",
    "CRMError",
    "NotifyError",
    "EmailError",
    "__version__",
]


def run_api():
    """启动 FastAPI 服务"""
    from .api import run
    run()

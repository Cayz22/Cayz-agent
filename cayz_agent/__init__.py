"""
cayz-agent：具备持久化记忆、联网搜索、知识库检索（RAG）、多模型支持与多 Agent 协作的企业级 Agent
"""

from .alerts import Alert, AlertLevel, check_alerts, get_alert_manager
from .config import Settings, get_settings, setup_logging
from .exceptions import (
    CayzAgentError,
    ConfigError,
    CRMError,
    EmailError,
    IntegrationError,
    LLMError,
    LLMRateLimitError,
    NotifyError,
    RAGConnectionError,
    RAGError,
    RAGIngestError,
    ToolError,
)
from .graph import AgentState, create_graph
from .integrations import (
    CRMClient,
    EmailSender,
    WeChatNotifier,
    get_crm_client,
    get_email_sender,
    get_notifier,
)
from .llm import create_llm, list_supported_providers
from .middleware import APIKeyAuthMiddleware, RateLimitMiddleware, setup_middleware
from .monitor import (
    export_prometheus,
    get_metrics_summary,
    get_registry,
    record_request,
    record_retry,
    record_route,
    record_token_usage,
    record_tool_call,
    record_validation_failure,
)
from .multi_agent import MultiAgentState, create_multi_agent_graph
from .retry import log_execution, retry_on_error
from .sanitizers import contains_harmful_content, sanitize_exception, sanitize_text
from .session import SessionInfo, SessionManager, get_session_manager
from .tools import AGENT_TOOLS
from .validators import InputValidationError, validate_search_query, validate_user_input

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
    "SessionInfo",
    "SessionManager",
    "get_session_manager",
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

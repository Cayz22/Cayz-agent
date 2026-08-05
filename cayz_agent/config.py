"""
集中配置管理 + 日志初始化

- 使用 pydantic-settings 从环境变量/.env 加载配置并做类型校验
- 提供 setup_logging() 统一配置标准 logging
- P2-9：通过 pydantic validator 在启动时 fail-fast 检测无效配置组合
"""

import logging
import sys

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置，自动从环境变量或 .env 文件读取"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== 多模型 Provider 配置 =====
    # 当前使用的 provider: openai / zhipu / qwen / ernie / ollama
    llm_provider: str = "openai"

    # 通用模型参数
    model_name: str = "qwen-turbo"
    temperature: float = 0.0
    # P0：LLM 请求超时（秒），防止单次慢请求拖垮可用性
    # OpenAI SDK 默认无超时，叠加同步调用会无限阻塞事件循环
    llm_request_timeout: float = 60.0

    # OpenAI 兼容服务（含通义千问 DashScope）
    openai_api_key: str = ""
    openai_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 智谱 GLM
    zhipu_api_key: str = ""
    zhipu_api_base: str = "https://open.bigmodel.cn/api/paas/v4"

    # 百度文心（通过 OpenAI 兼容代理）
    ernie_api_key: str = ""
    ernie_secret_key: str = ""
    ernie_api_base: str = "https://qianfan.baidubce.com/v2"

    # Ollama 本地推理
    ollama_base_url: str = "http://localhost:11434"

    # Tavily 联网搜索
    tavily_api_key: str = ""

    # 日志级别
    log_level: str = "INFO"

    # 持久化：memory（默认，重启丢失）或 sqlite（落盘）
    checkpoint_backend: str = "memory"
    sqlite_checkpoint_path: str = "checkpoints.db"

    # API 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ===== RAG 配置 =====
    # 向量数据库目录
    chroma_persist_dir: str = "./chroma_db"
    # Embedding 模型（OpenAI text-embedding-3-small 或本地模型）
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    # 文档切片大小（字符数）
    chunk_size: int = 500
    chunk_overlap: int = 50
    # 检索返回的最大文档数
    rag_top_k: int = 3

    # ===== 业务系统集成配置 =====
    # CRM 集成（true 使用模拟数据，false 对接真实 API）
    crm_use_mock: bool = True

    # 企业微信 Webhook 通知
    wecom_webhook_url: str = ""

    # SMTP 邮件发送
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_username: str = ""
    smtp_password: str = ""
    email_from_addr: str = ""
    smtp_use_ssl: bool = True

    # ===== API 安全配置 =====
    # API Key 鉴权（为空则不启用鉴权，仅开发环境允许为空）
    # 该 Key 拥有管理员权限（全部操作，包括删除知识库/会话）
    api_key: str = ""
    # 写权限 API Key（可上传/更新知识库，不可删除）。逗号分隔。
    # P1 权限分级：将"读+写"与"管理员（含删除）"分离，降低 Key 泄露影响面
    write_api_keys: str = ""
    # 只读 API Key（仅可对话与查询，不可修改知识库/会话）。逗号分隔。
    readonly_api_keys: str = ""
    # 是否强制要求鉴权（True 时 api_key 为空则非公开端点全部拒绝，防止生产裸奔）
    auth_required: bool = True
    # P0 安全加固：API Key 哈希化的服务端密钥（HMAC-SHA256）
    # 用于 hash_client_id() 防止 DB 泄露后离线预计算彩虹表。未配置时回退到无密钥 SHA256
    # 生产环境建议通过 API_KEY_HASH_SECRET 环境变量配置 32+ 字符随机串（如 secrets.token_urlsafe(32)）
    api_key_hash_secret: str = ""
    # 限流：每分钟每个客户端最大请求数（0 表示不限制）
    rate_limit_per_minute: int = 60
    # 写操作端点（知识库上传/删除/会话删除）的独立限流（每分钟每客户端，0 表示不额外限制）
    # P1 限流维度增强：对破坏性操作施加更严格限制
    rate_limit_write_per_minute: int = 20
    # 是否信任 X-Forwarded-For / X-Real-IP 头提取真实客户端 IP
    # P1 限流维度：反向代理（Nginx/ALB）后启用，否则所有限流命中代理 IP 形同虚设
    trust_forwarded_headers: bool = False
    # CORS 允许的来源（逗号分隔；* 表示全部允许但此时 credentials 强制为 False）
    # 生产环境应显式配置具体域名，如 https://your-domain.com
    cors_allowed_origins: str = "*"
    # 是否启用 API 文档（/docs /redoc /openapi.json），生产环境建议关闭
    docs_enabled: bool = False
    # M3 HTTPS 强制：是否将 HTTP 请求 301 重定向到 HTTPS
    # 生产环境应启用（配合反向代理 TLS 终止），开发环境关闭（本地 http:// 访问）
    force_https: bool = False
    # M4 请求体大小限制（字节），超过返回 413 Payload Too Large
    # 防 DoS：Pydantic 校验在 JSON 解析后，攻击者可发数 GB 请求体先耗尽内存
    # 默认 10MB，覆盖知识库上传（MAX_KNOWLEDGE_TEXT_LENGTH=100KB × MAX_BATCH_ITEMS=50 ≈ 5MB）
    max_request_body_size: int = 10 * 1024 * 1024
    # M4 Uvicorn 硬化：keep-alive 超时（秒），超时后关闭空闲连接
    # 防 Slowloris：攻击者保持大量半开连接耗尽连接池，短超时及时释放
    # 默认 5s（uvicorn 默认 5s），生产环境不建议调大
    uvicorn_timeout_keep_alive: int = 5
    # M4 Uvicorn 硬化：最大并发连接数，超过返回 503（uvicorn 层面拦截，早于应用处理）
    # 防连接耗尽：限制同时处理的请求数，避免单机被打满后无法响应正常请求
    # 0 表示不限制（uvicorn 默认），生产环境建议按机器规格设置（如 100-1000）
    uvicorn_limit_concurrency: int = 100
    # P2 知识库敏感检测：上传文档时扫描敏感信息（API Key/手机号/身份证等）
    # "warn"：检测到敏感信息仍上传但记录警告日志
    # "block"：检测到敏感信息则拒绝上传（返回 422）
    # "off"：不扫描
    knowledge_sensitive_scan: str = "warn"
    # P2 SMTP 收件人白名单：允许的邮箱域名（逗号分隔，如 example.com,company.cn）
    # 为空表示不限制（仅开发环境推荐，生产环境应配置白名单防止邮件外发泄露）
    smtp_allowed_domains: str = ""

    # ===== 日志配置 =====
    # 日志格式：text（传统文本）或 json（结构化，便于接入 ELK/Loki）
    log_format: str = "text"

    # ===== 告警后台 watcher 配置 =====
    # 是否启动后台告警检查线程（无需依赖 /metrics 端点被访问）
    alert_watcher_enabled: bool = True
    # 后台告警检查间隔（秒）
    alert_watcher_interval: int = 60

    # ===== P3 可观测性配置 =====
    # 是否启用 X-Request-ID 中间件（请求追踪）
    # 启用后：每个请求生成/透传 request_id，写入响应头与日志上下文
    request_id_enabled: bool = True
    # 优雅停机超时（秒）：等待在途请求完成的最长时间
    graceful_shutdown_timeout: int = 30

    # ===== P3 工具配置 =====
    # 工具访问的文件系统根目录（read_file/write_file 限制在此目录内，防越权）
    # 为空表示禁用文件工具；推荐配置为 /data/workspace 等专用目录
    tools_workspace_dir: str = ""
    # fetch_url 抓取超时（秒）
    tools_fetch_url_timeout: int = 15
    # fetch_url 最大响应体大小（字节，默认 1MB）
    tools_fetch_url_max_size: int = 1024 * 1024
    # fetch_url User-Agent
    tools_fetch_url_user_agent: str = "cayz-agent/1.0"
    # python_repl 执行超时（秒）
    tools_python_repl_timeout: int = 10
    # python_repl 最大输出字符数（防止超大输出打满 LLM 上下文）
    tools_python_repl_max_output: int = 4096

    # ===== P3 外部信息查询工具配置（可选，未配置则工具返回提示） =====
    # 和风天气 API（https://dev.qweather.com）
    weather_api_key: str = ""
    weather_api_base: str = "https://devapi.qweather.com/v7"
    # fixer.io 汇率 API（https://fixer.io）
    exchange_rate_api_key: str = ""
    exchange_rate_api_base: str = "https://data.fixer.io/api"
    # 百度翻译 API（https://fanyi-api.baidu.com）
    baidu_translate_app_id: str = ""
    baidu_translate_api_key: str = ""

    # ===== 缓存配置 =====
    # 是否启用 LLM 调用缓存（同 prompt 短期复用结果，节省 token）
    cache_llm_enabled: bool = True
    # 是否启用 Embedding 缓存（同文本短期复用向量，节省 API 调用）
    cache_embedding_enabled: bool = True
    # 是否启用 RAG 检索缓存（同 query 短期复用检索结果）
    cache_rag_search_enabled: bool = True
    # LLM 缓存最大条数
    cache_llm_maxsize: int = 256
    # LLM 缓存 TTL（秒），默认 5 分钟
    cache_llm_ttl: int = 300
    # Embedding 缓存最大条数
    cache_embedding_maxsize: int = 1024
    # Embedding 缓存 TTL（秒），默认 30 分钟
    cache_embedding_ttl: int = 1800
    # RAG 检索缓存最大条数
    cache_rag_maxsize: int = 128
    # RAG 检索缓存 TTL（秒），默认 2 分钟（知识库变更后短期失效）
    cache_rag_ttl: int = 120

    # ===== P2-9 启动时配置有效性校验（fail-fast）=====

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """temperature 必须在 [0.0, 2.0] 范围内"""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature 必须在 [0.0, 2.0]，当前值: {v}")
        return v

    @field_validator("cache_llm_maxsize", "cache_embedding_maxsize", "cache_rag_maxsize")
    @classmethod
    def _validate_cache_maxsize(cls, v: int) -> int:
        """缓存 maxsize 必须 >= 1，否则 TTLCache 行为异常"""
        if v < 1:
            raise ValueError(f"cache maxsize 必须 >= 1，当前值: {v}")
        return v

    @field_validator("rate_limit_per_minute", "rate_limit_write_per_minute")
    @classmethod
    def _validate_rate_limit(cls, v: int) -> int:
        """限流值必须 >= 0（0 表示不限制）"""
        if v < 0:
            raise ValueError(f"rate_limit 必须 >= 0，当前值: {v}")
        return v

    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        """llm_provider 必须是支持的值"""
        valid = {"openai", "zhipu", "qwen", "ernie", "ollama"}
        if v not in valid:
            raise ValueError(f"llm_provider 必须是 {valid} 之一，当前值: {v}")
        return v

    @model_validator(mode="after")
    def _validate_provider_key(self) -> "Settings":
        """P2-9：校验 provider 与 API Key 的一致性（仅 auth_required=True 时强制）

        - openai/qwen：需要 openai_api_key
        - zhipu：需要 zhipu_api_key 或 openai_api_key
        - ernie：需要 ernie_api_key
        - ollama：无需 key
        """
        if not self.auth_required:
            # 开发模式不强制校验，允许空 key 启动
            return self
        if self.llm_provider in ("openai", "qwen") and not self.openai_api_key:
            raise ValueError(f"provider={self.llm_provider} 需配置 OPENAI_API_KEY（auth_required=True 时强制）")
        if self.llm_provider == "zhipu" and not (self.zhipu_api_key or self.openai_api_key):
            raise ValueError("provider=zhipu 需配置 ZHIPU_API_KEY 或 OPENAI_API_KEY（auth_required=True 时强制）")
        if self.llm_provider == "ernie" and not self.ernie_api_key:
            raise ValueError("provider=ernie 需配置 ERNIE_API_KEY（auth_required=True 时强制）")
        return self


def get_settings() -> Settings:
    """获取配置单例。

    首次调用读取环境变量/.env 并实例化 Settings，后续调用返回同一实例。
    如需强制重新加载（例如测试中切换了 .env），调用 `reset_settings_cache()`。
    """
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()
    return _settings_singleton


def reset_settings_cache() -> None:
    """重置配置单例缓存（主要用于测试场景）"""
    global _settings_singleton
    _settings_singleton = None


# 模块级单例缓存
_settings_singleton: "Settings | None" = None


def setup_logging(level: str = "INFO", log_format: str = "text") -> None:
    """
    初始化标准 logging，统一格式与输出目标。

    Args:
        level: 日志级别（DEBUG / INFO / WARNING / ERROR）
        log_format: 输出格式 - "text"（传统文本）或 "json"（结构化，便于接入 ELK/Loki）

    应在应用入口（__main__ / web_app）启动时调用一次。
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    fmt = (log_format or "text").lower()

    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        handlers=[handler],
        force=True,  # 覆盖可能存在的已有配置
    )

    # P2 日志脱敏：在 root logger 上安装过滤器，确保所有日志输出前自动脱敏
    # P3 请求追踪：注入 request_id 到日志记录（JSON 格式时自动输出该字段）
    from .request_context import RequestIdLogFilter
    from .sanitizers import SanitizingLogFilter

    root_logger = logging.getLogger()
    # 避免重复添加过滤器（setup_logging 可能被多次调用）
    existing_filters = root_logger.filters
    if not any(isinstance(f, SanitizingLogFilter) for f in existing_filters):
        root_logger.addFilter(SanitizingLogFilter())
    if not any(isinstance(f, RequestIdLogFilter) for f in existing_filters):
        root_logger.addFilter(RequestIdLogFilter())

    # 降低第三方库的噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


class _JsonFormatter(logging.Formatter):
    """结构化 JSON 日志格式器

    输出字段：timestamp / level / logger / message / exc_info（如有）
    额外的 LogRecord 属性会自动合并到 JSON 中。
    """

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 合并 extra 字段（非标准属性）
        standard = set(
            vars(
                logging.LogRecord(
                    name="",
                    level=0,
                    pathname="",
                    lineno=0,
                    msg="",
                    args=(),
                    exc_info=None,
                )
            )
        )
        for key, value in vars(record).items():
            if key not in standard and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

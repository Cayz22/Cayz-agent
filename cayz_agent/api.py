"""
FastAPI REST API 服务

端点：
- GET  /health       深度健康检查（含依赖状态）
- GET  /metrics      Prometheus 指标导出
- POST /chat         同步对话（阻塞等待完整回复）
- POST /chat/stream  流式对话（SSE 逐 token 返回）

安全：
- API Key 鉴权（X-API-Key 或 Authorization: Bearer）
- 请求限流（滑动窗口，按 client_id 限流）
- CORS（可配置 allowed_origins）

错误响应规范：
- 输入验证失败：HTTP 422 + {"detail": "..."}
- 内部异常：HTTP 500 + {"detail": "..."}
- 鉴权失败：HTTP 401（由中间件处理）
- 限流：HTTP 429（由中间件处理）
"""

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel, Field

from . import __version__, app_state
from .alerts import check_alerts, start_alert_watcher, stop_alert_watcher
from .config import get_settings, setup_logging
from .graph import create_graph
from .middleware import _SCOPE_LEVEL, setup_middleware
from .monitor import (
    export_prometheus,
    get_metrics_summary,
    record_knowledge_delete,
    record_knowledge_upload,
    record_request,
    record_session_deleted,
    record_session_end,
    record_session_start,
)
from .sanitizers import detect_sensitive_info, sanitize_exception, sanitize_text
from .session import get_session_manager
from .validators import (
    MAX_BATCH_ITEMS,
    MAX_KNOWLEDGE_TEXT_LENGTH,
    InputValidationError,
    validate_knowledge_text,
    validate_thread_id,
    validate_user_input,
)

logger = logging.getLogger(__name__)

settings = get_settings()
setup_logging(settings.log_level, settings.log_format)


# ---- lifespan：替代已弃用的 on_event("startup"/"shutdown") ----


def _mask_secret(secret: str) -> str:
    """脱敏敏感字符串：显示前 4 位 + *** + 后 4 位，便于确认配置已加载而不泄露完整值

    短字符串（<=8 字符）全部替换为 ***，避免泄露大部分内容
    """
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}***{secret[-4:]}"


def _log_startup_report() -> None:
    """P3 启动配置自检报告：打印关键配置摘要（脱敏）+ 依赖预检状态

    目的：
    - 运维确认环境变量正确加载（API Key/数据库路径/限流等）
    - 快速发现配置不一致（如 provider=openai 但 OPENAI_API_KEY 未配置）
    - 不阻塞启动：依赖预检失败仅 WARN，由 /health/deep 暴露详情
    """
    from . import __version__

    logger.info("=" * 60)
    logger.info("cayz-agent v%s 启动配置自检报告", __version__)
    logger.info("=" * 60)

    # ---- LLM 配置 ----
    logger.info(
        "[LLM] provider=%s, model=%s, temperature=%.2f, timeout=%.0fs",
        settings.llm_provider,
        settings.model_name,
        settings.temperature,
        settings.llm_request_timeout,
    )
    key_map = {
        "openai": ("OPENAI_API_KEY", settings.openai_api_key),
        "zhipu": ("ZHIPU_API_KEY", settings.zhipu_api_key),
        "ernie": ("ERNIE_API_KEY", settings.ernie_api_key),
    }
    key_name, key_val = key_map.get(settings.llm_provider, ("(none)", ""))
    if settings.llm_provider == "ollama":
        logger.info("[LLM] ollama 模式，无需 API Key（base_url=%s）", settings.ollama_base_url)
    else:
        logger.info("[LLM] %s=%s", key_name, _mask_secret(key_val))

    # ---- 持久化 ----
    logger.info(
        "[Persistence] backend=%s, sqlite_path=%s", settings.checkpoint_backend, settings.sqlite_checkpoint_path
    )
    if settings.checkpoint_backend == "memory":
        logger.warning("[Persistence] 使用 memory 后端，重启后会话丢失（生产环境应用 sqlite）")

    # ---- RAG ----
    logger.info(
        "[RAG] embedding_provider=%s, model=%s, chunk_size=%d, top_k=%d",
        settings.embedding_provider,
        settings.embedding_model,
        settings.chunk_size,
        settings.rag_top_k,
    )
    logger.info("[RAG] chroma_persist_dir=%s", settings.chroma_persist_dir)

    # ---- 安全 ----
    logger.info(
        "[Security] auth_required=%s, api_key=%s, write_keys=%d, readonly_keys=%d",
        settings.auth_required,
        _mask_secret(settings.api_key) if settings.api_key else "<empty>",
        len([k for k in settings.write_api_keys.split(",") if k.strip()]),
        len([k for k in settings.readonly_api_keys.split(",") if k.strip()]),
    )
    logger.info(
        "[Security] rate_limit=%d/min, write_rate_limit=%d/min, force_https=%s",
        settings.rate_limit_per_minute,
        settings.rate_limit_write_per_minute,
        settings.force_https,
    )
    logger.info(
        "[Security] cors_origins=%s, trust_forwarded=%s, docs_enabled=%s",
        settings.cors_allowed_origins,
        settings.trust_forwarded_headers,
        settings.docs_enabled,
    )

    # ---- 可观测性 ----
    logger.info(
        "[Observability] log_level=%s, log_format=%s, request_id=%s",
        settings.log_level,
        settings.log_format,
        settings.request_id_enabled,
    )
    logger.info(
        "[Observability] alert_watcher=%s, alert_interval=%ds",
        settings.alert_watcher_enabled,
        settings.alert_watcher_interval,
    )

    # ---- 缓存 ----
    logger.info(
        "[Cache] llm=%s(%d/%ds), embedding=%s(%d/%ds), rag=%s(%d/%ds)",
        settings.cache_llm_enabled,
        settings.cache_llm_maxsize,
        settings.cache_llm_ttl,
        settings.cache_embedding_enabled,
        settings.cache_embedding_maxsize,
        settings.cache_embedding_ttl,
        settings.cache_rag_search_enabled,
        settings.cache_rag_maxsize,
        settings.cache_rag_ttl,
    )

    # ---- 集成 ----
    logger.info(
        "[Integration] crm_mock=%s, wecom_webhook=%s, smtp=%s:%d",
        settings.crm_use_mock,
        "<configured>" if settings.wecom_webhook_url else "<empty>",
        settings.smtp_host or "<empty>",
        settings.smtp_port,
    )

    # ---- 依赖预检（不阻塞启动，仅 WARN）----
    logger.info("-" * 60)
    logger.info("依赖预检：")
    deps = {
        "llm": _check_llm(),
        "chromadb": _check_chromadb(),
        "checkpointer": _check_checkpointer(),
    }
    for name, status in deps.items():
        if status["status"] == "healthy":
            logger.info("  ✓ %s: healthy", name)
        else:
            logger.warning("  ✗ %s: %s", name, status.get("detail", "unhealthy"))
    logger.info("=" * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时拉起后台告警 watcher 并标记就绪，关闭时优雅停机。

    P3 优雅停机流程（shutdown）：
    1. 立即标记 _ready=False：/health/ready 返回 503，负载均衡摘除流量
    2. 停止告警 watcher：避免后台线程在资源清理过程中误触发
    3. 执行注册的清理钩子（LIFO）：关闭 LangGraph SqliteSaver / ChromaDB 连接
    4. uvicorn 自身会等待在途请求完成（--graceful-timeout，应用层无需重复）

    放在 lifespan 而非模块导入时启动，避免测试导入 api 模块时
    意外拉起后台线程干扰测试隔离。
    """
    # ---- startup ----
    if settings.alert_watcher_enabled:
        start_alert_watcher(interval=float(settings.alert_watcher_interval))

    # P3：启动配置自检报告（脱敏输出，便于运维确认环境变量是否正确加载）
    _log_startup_report()

    # P3：注册资源清理钩子（shutdown 时 LIFO 执行）
    # LangGraph SqliteSaver 持久连接：通过 agent app 缓存的 conn 引用释放
    def _cleanup_agent_apps():
        global _agent_app, _agent_app_by_scope
        # 关闭每个 scope 的 graph 实例（含 SqliteSaver 连接）
        for scope, _graph in _agent_app_by_scope.items():
            try:
                # LangGraph CompiledGraph 不直接暴露 checkpointer，但 SqliteSaver
                # 在 GC 时会自动 close（__del__）。此处仅清除引用，让 GC 回收。
                _agent_app_by_scope[scope] = None
            except Exception:
                logger.exception("清理 scope=%s 的 agent app 失败", scope)
        _agent_app_by_scope.clear()
        _agent_app = None

    app_state.register_cleanup(_cleanup_agent_apps)

    # P3：标记就绪，开始接收流量
    app_state.set_ready(True)

    try:
        yield
    finally:
        # ---- shutdown ----
        # P3：立即标记未就绪，负载均衡探活摘除流量
        app_state.set_ready(False)

        # 停止后台 watcher
        stop_alert_watcher()

        # 执行资源清理钩子（带超时，避免 SIGTERM 后卡死）
        app_state.run_cleanups(timeout=float(settings.graceful_shutdown_timeout))


app = FastAPI(
    title="cayz-agent API",
    version=__version__,
    description="具备持久化记忆、联网搜索与安全护栏的 Agent API",
    # 生产环境关闭 API 文档，防止端点结构泄露
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
    lifespan=lifespan,
)


# ---- 全局异常处理（兜底端点 try/except 之外的未捕获异常）----


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic 请求体校验失败：统一为 422 + 标准 detail 格式。

    FastAPI 默认返回 422 但 detail 是 errors 数组，此处统一为字符串消息，
    与端点内 InputValidationError 抛出的 422 保持一致格式。
    """
    try:
        from .monitor import record_request as _rec  # noqa: F401

        _rec(request_type=request.url.path, success=False, latency=0.0)
    except Exception:
        pass
    # 把 errors 列表序列化为可读字符串
    errors_str = (
        "; ".join(f"{'.'.join(str(p) for p in e.get('loc', []))}: {e.get('msg', '')}" for e in exc.errors())
        or "请求参数校验失败"
    )
    return JSONResponse(
        status_code=422,
        content={"detail": f"输入无效: {errors_str}"},
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    """兜底未捕获异常：统一为 500 + 通用错误消息。

    P2 错误响应体收敛：
    - 任何端点 try/except 之外的异常都由此处理
    - HTTPException 直接放行，由 FastAPI 默认处理器返回（保持 status_code 与 detail）
    - 其他异常：开发模式（auth_required=False）返回脱敏后的错误详情便于调试；
      生产模式（auth_required=True）仅返回通用消息，隐藏内部实现细节与堆栈
    """
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    logger.exception("未捕获的异常 (path=%s): %s", request.url.path, exc)
    # P2：生产环境仅返回通用消息，避免泄露框架/库/路径等内部实现细节
    if settings.auth_required:
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误，请稍后重试或联系管理员"},
        )
    # 开发环境返回脱敏后的详情便于调试
    return JSONResponse(
        status_code=500,
        content={"detail": f"内部错误: {sanitize_exception(exc)}"},
    )


# ---- 中间件注册（FastAPI LIFO 执行：后注册的先执行）----
# 注册顺序：鉴权/限流 → CORS
# 这样 CORS 中间件在执行链最外层，所有响应（包括 401/429）都带 CORS 头，
# 浏览器才能读取错误响应体。
setup_middleware(app)

# ---- CORS ----
# 解析允许的来源；当配置为 * 时强制关闭 credentials（符合 CORS 规范，防止 CSRF）
_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
_is_wildcard = _origins == ["*"]
if _is_wildcard:
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "CORS 配置为 *（通配符），已强制关闭 credentials。" "生产环境应通过 CORS_ALLOWED_ORIGINS 显式指定域名。"
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=not _is_wildcard,  # 通配符时不允许携带凭证
    allow_methods=["*"],
    allow_headers=["*"],
)

# Agent 图（延迟初始化，避免模块导入时触发 LLM/ChromaDB 连接）
# 测试可通过 patch("cayz_agent.api._agent_app", ...) 直接覆盖。
_agent_app = None
# P0 工具权限分级：按 scope 缓存图实例
_agent_app_by_scope: dict = {}
# P1 修复：单例创建使用双重检查锁定，避免首批并发请求创建多实例
# （多实例会创建多个 SqliteSaver 连接，加剧 SQLite 写冲突）
import threading as _threading

_agent_app_lock = _threading.Lock()


def get_agent_app():
    """获取默认 Agent 图单例（admin scope，向后兼容）

    P1 修复：使用双重检查锁定（double-checked locking）保证线程安全。
    """
    global _agent_app
    if _agent_app is None:
        with _agent_app_lock:
            if _agent_app is None:
                _agent_app = create_graph("admin")
    return _agent_app


def get_agent_app_for_scope(scope: str):
    """获取指定 scope 的 Agent 图实例（P0 工具权限分级）

    P1 修复：使用 dict 操作的原子性 + 锁保护创建过程。
    Python 的 dict get/set 是原子的（GIL），但创建 graph 的过程不是，
    需锁保护避免重复创建。
    """
    if scope not in _agent_app_by_scope:
        with _agent_app_lock:
            if scope not in _agent_app_by_scope:
                _agent_app_by_scope[scope] = create_graph(scope)
    return _agent_app_by_scope[scope]


# ---- P1 权限分级依赖 ----
# require_scope("write") / require_scope("admin") 作为端点 Depends，
# 校验当前请求的 scope 等级是否足够。readonly < write < admin。


def _get_request_scope(request: Request) -> str:
    """从 request.state 读取 scope（由鉴权中间件写入）。

    开发模式（无 Key）下中间件已写入 "admin"，故此处总有值。
    """
    return getattr(request.state, "scope", "admin")


def _get_request_client_id(request: Request) -> str:
    """从 request.state 读取客户端标识（API Key 或真实 IP），用于会话归属。"""
    return getattr(request.state, "client_id", "unknown")


def require_scope(min_scope: str):
    """生成端点权限校验依赖。

    Args:
        min_scope: 该端点所需的最低权限等级（"readonly" / "write" / "admin"）

    用法：
        @app.post("/knowledge/upload")
        async def upload(_: None = Depends(require_scope("write"))):
            ...
    """
    required_level = _SCOPE_LEVEL[min_scope]

    def _checker(request: Request):
        actual = _SCOPE_LEVEL.get(_get_request_scope(request), 0)
        if actual < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"权限不足：需要 {min_scope} 及以上权限",
            )
        return None

    return _checker


def _scan_knowledge_sensitive(text: str, source: str = ""):
    """P2 知识库敏感检测：扫描文档内容并按配置策略处理。

    - "off"：跳过扫描
    - "warn"：记录警告日志但允许上传
    - "block"：抛出 422 拒绝上传

    Raises:
        HTTPException: 当 mode="block" 且检测到敏感信息时
    """
    mode = settings.knowledge_sensitive_scan.lower()
    if mode == "off":
        return
    found = detect_sensitive_info(text)
    if not found:
        return
    msg = f"知识库文档包含敏感信息类型: {found}"
    if source:
        msg += f"（source={source}）"
    logger.warning(msg)
    if mode == "block":
        raise HTTPException(
            status_code=422,
            detail=f"文档包含敏感信息（{', '.join(found)}），已拒绝上传。请脱敏后重试。",
        )


# ---- 请求/响应模型 ----


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    thread_id: str | None = Field(None, description="会话 ID，不传则自动生成")


class ChatResponse(BaseModel):
    reply: str = Field(..., description="Agent 回复")
    thread_id: str = Field(..., description="会话 ID")


# ---- 依赖检查 ----


def _check_llm() -> dict:
    """检查 LLM provider 配置是否就绪"""
    try:
        from .llm import list_supported_providers

        provider = settings.llm_provider
        if provider not in list_supported_providers():
            return {"status": "unhealthy", "detail": f"未知 provider: {provider}"}
        # 检查对应 API Key 是否配置（ollama 无需 key）
        if provider == "ollama":
            return {"status": "healthy"}
        key_map = {
            "openai": settings.openai_api_key,
            "qwen": settings.openai_api_key,
            "zhipu": settings.zhipu_api_key,
            "ernie": settings.ernie_api_key,
        }
        key = key_map.get(provider, "")
        if not key:
            return {"status": "unhealthy", "detail": f"{provider} API Key 未配置"}
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}


def _check_chromadb() -> dict:
    """检查 ChromaDB 是否可用"""
    try:
        from .rag import get_rag_manager

        manager = get_rag_manager()
        # count() 触发底层集合访问，验证可用性
        _ = manager.count()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}


def _check_checkpointer() -> dict:
    """检查持久化后端是否可用"""
    try:
        backend = settings.checkpoint_backend.lower()
        if backend == "sqlite":
            import sqlite3

            conn = sqlite3.connect(settings.sqlite_checkpoint_path)
            conn.close()
            return {"status": "healthy", "backend": "sqlite"}
        return {"status": "healthy", "backend": "memory"}
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}


# ---- 端点 ----


@app.get("/health")
async def health():
    """轻量健康检查（公开端点，仅返回存活状态，用于 Docker healthcheck / 负载均衡探活）"""
    return {"status": "ok", "version": __version__}


@app.get("/health/ready")
async def health_ready():
    """P3 就绪探针：检查应用是否就绪接收流量（公开端点，用于 K8s readinessProbe）。

    与 /health 的区别：
    - /health（存活）：进程存活即返回 200，K8s livenessProbe 用
    - /health/ready（就绪）：依赖初始化完成且 _ready 标志为 True 才返回 200，
      K8s readinessProbe 用；未就绪时返回 503，Service 摘除流量

    与 /health/deep 的区别：
    - /health/deep：返回依赖详情（需鉴权，运维排查用）
    - /health/ready：仅返回就绪状态（公开，K8s 探针用，无敏感信息）
    """
    ready = app_state.is_ready()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "version": __version__,
            "uptime_seconds": round(time.time() - app_state.get_app_started_at(), 2),
        },
    )


@app.get("/health/deep")
async def health_deep():
    """深度健康检查：包含依赖组件状态与监控指标摘要（需鉴权）"""
    deps = {
        "llm": _check_llm(),
        "chromadb": _check_chromadb(),
        "checkpointer": _check_checkpointer(),
    }
    all_healthy = all(d["status"] == "healthy" for d in deps.values())
    return {
        "status": "ok" if all_healthy else "degraded",
        "version": __version__,
        "dependencies": deps,
        "metrics": get_metrics_summary(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus 指标导出（需鉴权，防止业务指标泄露）"""
    # 触发一次告警检查
    check_alerts()
    return PlainTextResponse(export_prometheus(), media_type="text/plain; version=0.0.4")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """同步对话：等待 Agent 完成后返回完整回复"""
    start = time.perf_counter()
    # 输入验证：失败返回 422
    try:
        clean_message = validate_user_input(req.message)
    except InputValidationError as e:
        record_request(request_type="chat", success=False, latency=time.perf_counter() - start)
        raise HTTPException(status_code=422, detail=f"输入无效: {e}")

    # M1 会话 ID：未传则用 secrets.token_urlsafe(32) 加密安全生成；
    # 传入则做格式校验（防日志注入 / 超长 DoS / 特殊字符）
    try:
        tid = validate_thread_id(req.thread_id) if req.thread_id else f"api-{secrets.token_urlsafe(32)}"
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=f"输入无效: {e}")

    # P0 IDOR 修复：非管理员若传入已存在的 thread_id，必须校验归属，防止越权读取/接管他人会话
    manager = get_session_manager()
    client_id = _get_request_client_id(request)
    scope = _get_request_scope(request)
    if req.thread_id and scope != "admin":
        if manager.session_exists(tid) and not manager.owns_session(tid, client_id):
            # 不区分「不存在」与「不归属」，避免泄露存在性
            raise HTTPException(status_code=403, detail="无权访问该会话")

    config = {"configurable": {"thread_id": tid}}

    # 记录会话活跃时间（用于过期清理）+ P1 IDOR：记录会话归属
    manager.touch_session(tid, owner=client_id)

    record_session_start()
    try:
        result = await asyncio.to_thread(
            get_agent_app_for_scope(scope).invoke,
            {"messages": [HumanMessage(content=clean_message)]},
            config,
        )
        ai_message = result["messages"][-1].content
        safe_reply = sanitize_text(ai_message)
        record_request(request_type="chat", success=True, latency=time.perf_counter() - start)
        return ChatResponse(reply=safe_reply, thread_id=tid)

    except Exception as e:
        record_request(request_type="chat", success=False, latency=time.perf_counter() - start)
        # P2 错误响应体收敛：生产模式返回通用消息，开发模式返回脱敏详情
        if settings.auth_required:
            raise HTTPException(status_code=500, detail="Agent 执行出错，请稍后重试")
        raise HTTPException(status_code=500, detail=f"Agent 执行出错: {sanitize_exception(e)}")

    finally:
        record_session_end()


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """流式对话：以 SSE 逐 token 返回 Agent 回复"""
    start = time.perf_counter()

    # 输入验证：失败返回 422（不进入流式）
    try:
        clean_message = validate_user_input(req.message)
    except InputValidationError as e:
        record_request(request_type="stream", success=False, latency=time.perf_counter() - start)
        raise HTTPException(status_code=422, detail=f"输入无效: {e}")

    # M1 会话 ID：未传则用 secrets.token_urlsafe(32) 加密安全生成；
    # 传入则做格式校验（防日志注入 / 超长 DoS / 特殊字符）
    try:
        tid = validate_thread_id(req.thread_id) if req.thread_id else f"api-{secrets.token_urlsafe(32)}"
    except InputValidationError as e:
        raise HTTPException(status_code=422, detail=f"输入无效: {e}")

    # P0 IDOR 修复：非管理员若传入已存在的 thread_id，必须校验归属
    manager = get_session_manager()
    client_id = _get_request_client_id(request)
    scope = _get_request_scope(request)
    if req.thread_id and scope != "admin":
        if manager.session_exists(tid) and not manager.owns_session(tid, client_id):
            raise HTTPException(status_code=403, detail="无权访问该会话")

    # 记录会话活跃时间（与 /chat 保持一致，防止被 cleanup_expired_sessions 误删）
    # P1 IDOR：记录会话归属
    manager.touch_session(tid, owner=client_id)

    config = {"configurable": {"thread_id": tid}}

    async def _stream():
        # 流式会话生命周期埋点：与 /chat 对齐，保证 active_sessions 指标准确
        record_session_start()
        try:
            raw = ""
            # P0 修复：使用 astream 异步生成器，避免同步 .stream() 阻塞事件循环
            async for chunk, _metadata in get_agent_app_for_scope(scope).astream(
                {"messages": [HumanMessage(content=clean_message)]},
                config=config,
                stream_mode="messages",
            ):
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    raw += chunk.content
                    # 实时脱敏：在每个 chunk 发出前先 sanitize，避免敏感信息已发到客户端
                    safe_chunk = sanitize_text(chunk.content)
                    yield f"data: {json.dumps({'token': safe_chunk, 'thread_id': tid}, ensure_ascii=False)}\n\n"
            # 流结束后发送脱敏后的完整文本（覆盖跨 chunk 的敏感信息）
            safe = sanitize_text(raw)
            record_request(request_type="stream", success=True, latency=time.perf_counter() - start)
            yield f"data: {json.dumps({'done': True, 'reply': safe, 'thread_id': tid}, ensure_ascii=False)}\n\n"
        except Exception as e:
            record_request(request_type="stream", success=False, latency=time.perf_counter() - start)
            yield f"data: {json.dumps({'error': sanitize_exception(e), 'thread_id': tid}, ensure_ascii=False)}\n\n"
        finally:
            # 无论成功/失败都释放 active_sessions
            record_session_end()

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---- 会话管理端点 ----
# P1 IDOR 修复：非管理员仅能访问/删除自己的会话（按 client_id 归属过滤）


@app.get("/sessions")
async def list_sessions(limit: int = 100, offset: int = 0, request: Request = None):
    """列出会话（admin 查全部，其他用户仅查自己的）

    P1 分页修复：返回真实 total（符合条件的总会话数），而非当前页 len(sessions)。
    """
    start = time.perf_counter()
    manager = get_session_manager()
    # admin 无 owner 过滤；非 admin 按 client_id 过滤
    owner_filter = None if _get_request_scope(request) == "admin" else _get_request_client_id(request)
    sessions, total = manager.list_sessions(limit=limit, offset=offset, owner_filter=owner_filter)
    record_request(request_type="sessions_list", success=True, latency=time.perf_counter() - start)
    return {
        "sessions": [s.to_dict() for s in sessions],
        "total": total,
    }


@app.get("/sessions/{thread_id}")
async def get_session_detail(thread_id: str, request: Request = None):
    """获取指定会话详情（admin 可查任意，其他用户仅查自己的）"""
    start = time.perf_counter()
    manager = get_session_manager()
    owner_filter = None if _get_request_scope(request) == "admin" else _get_request_client_id(request)
    info = manager.get_session(thread_id, owner_filter=owner_filter)
    record_request(request_type="sessions_get", success=True, latency=time.perf_counter() - start)
    if info is None:
        return {"exists": False, "thread_id": thread_id}
    return info


@app.delete("/sessions/{thread_id}")
async def delete_session(thread_id: str, request: Request = None):
    """删除指定会话（admin 可删任意，其他用户仅删自己的）"""
    start = time.perf_counter()
    manager = get_session_manager()
    owner_filter = None if _get_request_scope(request) == "admin" else _get_request_client_id(request)
    deleted = manager.delete_session(thread_id, owner_filter=owner_filter)
    if deleted:
        record_session_deleted()
    record_request(request_type="sessions_delete", success=deleted, latency=time.perf_counter() - start)
    return {"deleted": deleted, "thread_id": thread_id}


# ---- 知识库管理端点 ----


class KnowledgeUploadRequest(BaseModel):
    text: str = Field(..., max_length=MAX_KNOWLEDGE_TEXT_LENGTH, description="要上传的文本内容")
    source: str = Field("api_upload", max_length=200, description="文档来源标识")


class KnowledgeBatchUploadRequest(BaseModel):
    items: list[KnowledgeUploadRequest] = Field(
        ..., max_length=MAX_BATCH_ITEMS, description=f"批量文档列表（最多 {MAX_BATCH_ITEMS} 条）"
    )


class KnowledgeUpdateRequest(BaseModel):
    source: str = Field(..., max_length=200, description="要更新的文档来源")
    text: str = Field(..., max_length=MAX_KNOWLEDGE_TEXT_LENGTH, description="新的文档内容")


@app.get("/knowledge/sources")
async def list_knowledge_sources():
    """列出知识库中所有文档来源"""
    start = time.perf_counter()
    from .rag import get_rag_manager

    manager = get_rag_manager()
    sources = manager.list_sources()
    record_request(request_type="knowledge_sources", success=True, latency=time.perf_counter() - start)
    return {"sources": sources}


@app.get("/knowledge/count")
async def knowledge_count():
    """获取知识库文档片段总数"""
    start = time.perf_counter()
    from .rag import get_rag_manager

    manager = get_rag_manager()
    count = manager.count()
    record_request(request_type="knowledge_count", success=True, latency=time.perf_counter() - start)
    return {"count": count}


@app.post("/knowledge/upload")
async def knowledge_upload(req: KnowledgeUploadRequest, _: None = Depends(require_scope("write"))):
    """上传文档到知识库（需 write 及以上权限）"""
    start = time.perf_counter()
    from .rag import get_rag_manager

    try:
        clean_text = validate_knowledge_text(req.text)
    except InputValidationError as e:
        record_request(request_type="knowledge_upload", success=False, latency=time.perf_counter() - start)
        raise HTTPException(status_code=422, detail=str(e))

    # P2 知识库敏感检测：扫描文档中的敏感信息
    _scan_knowledge_sensitive(clean_text, source=req.source)

    manager = get_rag_manager()
    count = manager.add_documents(clean_text, source=req.source)
    if count > 0:
        record_knowledge_upload(count)
    record_request(request_type="knowledge_upload", success=count > 0, latency=time.perf_counter() - start)
    return {"success": count > 0, "chunks": count, "source": req.source}


@app.post("/knowledge/batch-upload")
async def knowledge_batch_upload(req: KnowledgeBatchUploadRequest, _: None = Depends(require_scope("write"))):
    """批量上传文档到知识库（原子性：中途失败时回滚已成功插入的 source；需 write 及以上权限）"""
    start = time.perf_counter()
    from .rag import get_rag_manager

    # 逐条验证（Pydantic 已做 max_length 拦截，这里补充空文本检查 + P2 敏感检测）
    validated_items = []
    rejected = []
    for idx, item in enumerate(req.items):
        try:
            clean_text = validate_knowledge_text(item.text)
            # P2 知识库敏感检测
            _scan_knowledge_sensitive(clean_text, source=item.source)
            validated_items.append({"text": clean_text, "source": item.source})
        except InputValidationError as e:
            rejected.append({"index": idx, "source": item.source, "error": str(e)})
        except HTTPException as e:
            # 敏感检测 block 模式抛出的 422
            rejected.append({"index": idx, "source": item.source, "error": e.detail})

    manager = get_rag_manager()

    # 逐条入库并跟踪已成功 source 及其新增 chunk ID（用于失败时精确回滚）
    # P1 修复：旧实现按 source 回滚会误删该 source 下已有的历史片段，
    # 现在按本次新增的 chunk ID 精确回滚
    succeeded_items: list[dict] = []  # [{"source": str, "ids": list[str], "count": int}]
    total = 0
    batch_failed = False
    batch_error: str | None = None

    for item in validated_items:
        try:
            new_ids = manager.add_documents_returning_ids(item["text"], source=item["source"])
            if new_ids:
                total += len(new_ids)
                succeeded_items.append({"source": item["source"], "ids": new_ids, "count": len(new_ids)})
        except Exception as e:
            # 中途失败：按本次新增的 chunk ID 精确回滚，避免误删历史片段
            batch_failed = True
            batch_error = sanitize_exception(e)
            logger.warning(
                "批量上传中途失败 (source=%s): %s，开始回滚 %d 个已成功条目",
                item["source"],
                batch_error,
                len(succeeded_items),
            )
            for succ in succeeded_items:
                try:
                    rollback_count = manager.delete_by_ids(succ["ids"])
                    logger.info(
                        "回滚 source=%s，删除 %d 个本次新增片段（历史片段保留）",
                        succ["source"],
                        rollback_count,
                    )
                except Exception as rollback_err:
                    logger.error(
                        "回滚 source=%s 失败: %s（需人工清理）",
                        succ["source"],
                        sanitize_exception(rollback_err),
                    )
            total = 0
            break

    if total > 0:
        record_knowledge_upload(total)
    record_request(
        request_type="knowledge_batch_upload",
        success=total > 0 and not batch_failed,
        latency=time.perf_counter() - start,
    )
    return {
        "success": total > 0 and not batch_failed,
        "total_chunks": total,
        "doc_count": len(validated_items),
        "rejected": rejected,
        "rolled_back": batch_failed,
        "error": batch_error,
    }


@app.put("/knowledge/update")
async def knowledge_update(req: KnowledgeUpdateRequest, _: None = Depends(require_scope("write"))):
    """更新知识库文档（按 source 替换；需 write 及以上权限）"""
    start = time.perf_counter()
    from .rag import get_rag_manager

    try:
        clean_text = validate_knowledge_text(req.text)
    except InputValidationError as e:
        record_request(request_type="knowledge_update", success=False, latency=time.perf_counter() - start)
        raise HTTPException(status_code=422, detail=str(e))

    # P2 知识库敏感检测
    _scan_knowledge_sensitive(clean_text, source=req.source)

    manager = get_rag_manager()
    count = manager.update_document(req.source, clean_text)
    if count > 0:
        record_knowledge_upload(count)
    record_request(request_type="knowledge_update", success=count > 0, latency=time.perf_counter() - start)
    return {"success": count > 0, "chunks": count, "source": req.source}


@app.delete("/knowledge/{source}")
async def knowledge_delete(source: str, _: None = Depends(require_scope("admin"))):
    """按来源删除知识库文档（需 admin 权限）"""
    start = time.perf_counter()
    from .rag import get_rag_manager

    manager = get_rag_manager()
    deleted = manager.delete_by_source(source)
    if deleted > 0:
        record_knowledge_delete(deleted)
    record_request(request_type="knowledge_delete", success=True, latency=time.perf_counter() - start)
    return {"deleted": deleted, "source": source}


def run():
    """启动 API 服务（供 CLI 调用）

    M4 服务器硬化参数：
    - timeout_keep_alive：空闲 keep-alive 连接超时秒数，防 Slowloris 慢速攻击
    - limit_concurrency：最大并发连接数，超限返回 503，防连接耗尽 DoS
      （仅当 > 0 时传给 uvicorn，0 表示不限制）
    """
    import uvicorn

    uvicorn.run(
        "cayz_agent.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        timeout_keep_alive=settings.uvicorn_timeout_keep_alive,
        **({"limit_concurrency": settings.uvicorn_limit_concurrency} if settings.uvicorn_limit_concurrency > 0 else {}),
    )

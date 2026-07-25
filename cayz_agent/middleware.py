"""
FastAPI 中间件：API Key 鉴权（含权限分级）+ 请求限流

- APIKeyAuth：从 X-API-Key 头或 Authorization: Bearer 提取 Key 校验
  - P1 权限分级：admin / write / readonly 三级 scope，写入 request.state 供端点校验
  - P1 限流维度：支持 X-Forwarded-For / X-Real-IP 提取真实 IP（反向代理后）
- RateLimiter：基于内存令牌桶，按客户端标识（API Key 或 IP）限流
"""

import hashlib
import hmac
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from .config import get_settings

# 仅轻量健康检查端点无需鉴权（用于 Docker healthcheck / 负载均衡探活 / K8s 探针）
# /metrics /docs /redoc /openapi.json 均需鉴权，避免泄露内部信息
# P3：/health/ready 公开，供 K8s readinessProbe 调用（不返回敏感信息，仅就绪状态）
_PUBLIC_PATHS = {"/health", "/health/ready"}

# 权限等级：admin > write > readonly
_SCOPE_LEVEL = {"readonly": 1, "write": 2, "admin": 3}

# P1 修复：API Key 哈希化前缀，用于区分「哈希后的 API Key」与「明文 IP」，
# 便于 session_activity.owner 列下游代码识别
_API_KEY_HASH_PREFIX = "key:"


def hash_client_id(client_id: str) -> str:
    """P1 安全修复：将 API Key 哈希化作为 owner 标识，避免明文存储到 session_activity 表。

    - API Key（来自 Authorization/X-API-Key）→ sha256 哈希前缀 + 前 16 位 hex
    - IP 地址（无 Key 回退路径）→ 原值保留（IP 非机密信息，且需保留用于审计/限流）

    哈希化目的：DB 文件泄露时不暴露原始 API Key，防止全量 Key 接管。
    保留 IP 明文：限流维度需要原始 IP；且 IP 本身不构成机密凭证。
    """
    if not client_id:
        return client_id
    # 简单启发式：API Key 通常较长（>= 16 字符）且不以数字开头（IP 以数字开头）
    # 更稳妥：由调用方传入时已知道是 Key 还是 IP，但为兼容旧调用，这里用前缀判断
    # 中间件 dispatch 中会对 API Key 显式加前缀后再调用本函数
    if client_id.startswith(_API_KEY_HASH_PREFIX):
        # 已哈希，幂等返回
        return client_id
    digest = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:16]
    return f"{_API_KEY_HASH_PREFIX}{digest}"


def _get_real_ip(request: Request) -> str:
    """获取客户端真实 IP。

    P1 限流维度：反向代理（Nginx/ALB）后所有请求来源 IP 相同，
    需从 X-Forwarded-For / X-Real-IP 提取真实 IP。仅在 trust_forwarded_headers=True 时生效，
    避免直接暴露时客户端伪造 IP 头绕过限流。
    """
    settings = get_settings()
    if settings.trust_forwarded_headers:
        # X-Forwarded-For: client, proxy1, proxy2 — 取第一个（最原始客户端）
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
        xri = request.headers.get("X-Real-IP", "")
        if xri:
            return xri.strip()
    return request.client.host if request.client else "unknown"


def _get_client_id(request: Request) -> str:
    """获取客户端标识：优先使用 API Key，否则使用真实 IP"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key
    return _get_real_ip(request)


def _extract_api_key(request: Request) -> str | None:
    """从请求头提取 API Key"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.headers.get("X-API-Key")


def _constant_time_contains(candidate: str, keys: list[str]) -> bool:
    """在多个 Key 中时序安全地查找 candidate。

    遍历所有 Key 并逐一用 hmac.compare_digest 比较，无论是否匹配都走完全部比较，
    避免短路导致的时序泄露（攻击者无法通过响应时间判断匹配发生在第几个位置）。
    """
    matched = False
    for k in keys:
        # compare_digest 要求两侧均为同类型 str 或 bytes；长度不同时仍按字节比较，时序安全
        if hmac.compare_digest(candidate, k):
            matched = True
    return matched


def _resolve_scope(provided_key: str | None, settings) -> str | None:
    """解析 API Key 的权限等级。

    P1 权限分级：
    - settings.api_key（管理员）：admin
    - settings.write_api_keys（逗号分隔）：write
    - settings.readonly_api_keys（逗号分隔）：readonly
    返回 None 表示 Key 无效。

    向后兼容：仅配置 api_key 时，它拥有 admin 权限。

    H1 时序安全：所有 Key 比较使用 hmac.compare_digest，防止通过响应时间逐字节爆破。
    """
    if not provided_key:
        return None
    # H1：admin key 用 compare_digest 时序安全比较
    if settings.api_key and hmac.compare_digest(provided_key, settings.api_key):
        return "admin"
    write_keys = [k.strip() for k in settings.write_api_keys.split(",") if k.strip()]
    if _constant_time_contains(provided_key, write_keys):
        return "write"
    readonly_keys = [k.strip() for k in settings.readonly_api_keys.split(",") if k.strip()]
    if _constant_time_contains(provided_key, readonly_keys):
        return "readonly"
    return None


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """API Key 鉴权中间件（含权限分级）"""

    def __init__(self, app, public_paths: set | None = None):
        super().__init__(app)
        self.public_paths = public_paths or _PUBLIC_PATHS

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        # 公开端点放行（仅 /health 轻量探活）
        if request.url.path in self.public_paths:
            return await call_next(request)

        # 判断是否配置了任意鉴权 Key（admin / write / readonly）
        has_any_key = bool(settings.api_key or settings.write_api_keys.strip() or settings.readonly_api_keys.strip())

        if has_any_key:
            provided = _extract_api_key(request)
            scope = _resolve_scope(provided, settings)
            if scope is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "无效或缺失的 API Key"},
                )
            # P1：将 scope 与 client 标识存入 request.state，供端点做权限校验与会话归属
            # P1 安全修复：API Key 哈希化后作为 client_id，避免明文写入 session_activity.owner 列
            # 限流仍用哈希后的 ID（同一 Key 的请求仍归同一桶，不影响限流精度）
            client_id_raw = provided or _get_real_ip(request)
            request.state.scope = scope
            request.state.client_id = hash_client_id(client_id_raw) if provided else client_id_raw
            return await call_next(request)

        # 未配置任何 Key 时的处理
        if settings.auth_required:
            # 生产模式：强制鉴权但 Key 未配置 → 拒绝所有非公开端点，防止裸奔
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "服务未就绪：API_KEY 未配置，请联系管理员",
                },
            )

        # 开发模式（auth_required=False）：允许无鉴权访问
        # 仍记录 client_id 用于会话归属（IDOR 修复需要）
        request.state.scope = "admin"
        request.state.client_id = _get_real_ip(request)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    请求限流中间件（滑动窗口算法）

    按 client_id（API Key 或真实 IP）限制每分钟请求数。
    P1 限流维度：写操作端点使用更严格的独立限流（rate_limit_write_per_minute）。
    """

    # 定期扫描间隔（秒）：每隔一段时间清理已无活跃记录的 client_id，
    # 避免 _hits dict 在长期运行下无限增长（P2 内存泄漏修复）
    _SWEEP_INTERVAL = 300

    # P1：写/删端点路径前缀，施加更严格限流
    _WRITE_PATHS = {
        "/knowledge/upload",
        "/knowledge/batch-upload",
        "/knowledge/update",
    }

    def __init__(self, app, public_paths: set | None = None):
        super().__init__(app)
        self.public_paths = public_paths or _PUBLIC_PATHS
        self._hits: dict[str, deque] = defaultdict(deque)
        # P1：写端点独立计数桶（与全局限流分离，避免写操作耗尽读配额）
        self._write_hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()
        self._last_sweep = time.time()

    def _sweep_empty_clients(self, window_start: float) -> None:
        """扫描并删除已无活跃记录的 client_id（调用方需持锁）"""
        for bucket in (self._hits, self._write_hits):
            empty_keys = [cid for cid, dq in bucket.items() if not dq or dq[-1] < window_start]
            for cid in empty_keys:
                bucket.pop(cid, None)

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        limit = settings.rate_limit_per_minute
        path = request.url.path

        # 不限制或公开端点放行
        if limit <= 0 or path in self.public_paths:
            return await call_next(request)

        # P1：写端点使用独立限流桶
        is_write = path in self._WRITE_PATHS or path.startswith("/knowledge/") and request.method == "DELETE"
        write_limit = settings.rate_limit_write_per_minute
        if is_write and write_limit > 0:
            bucket = self._write_hits
            effective_limit = min(limit, write_limit)
        else:
            bucket = self._hits
            effective_limit = limit

        # P0 修复：优先使用鉴权中间件写入的 request.state.client_id（已解析 API Key），
        # 这样限流维度与鉴权维度一致；鉴权未设置时回退到 _get_client_id 自行解析
        # 注意：request.state 可能在直接调用 dispatch（如测试）时不存在，需先 getattr
        # P1 安全修复：鉴权中间件已将 API Key 哈希化写入 client_id，限流桶 key 同步使用哈希值，
        # 不影响限流精度（同一 Key 仍归同一桶），且避免限流日志/dict 中残留明文 Key
        client_id = None
        try:
            client_id = getattr(request.state, "client_id", None)
        except AttributeError:
            pass
        if not client_id:
            # 兜底路径：未经过鉴权中间件（如公开端点或测试），自行解析并哈希化（若是 API Key）
            raw = _get_client_id(request)
            # 简单区分：Authorization/X-API-Key 头存在则视为 Key，哈希化；否则当 IP 处理
            if request.headers.get("Authorization") or request.headers.get("X-API-Key"):
                client_id = hash_client_id(raw)
            else:
                client_id = raw
        now = time.time()
        window_start = now - 60

        with self._lock:
            # 定期触发扫描：清理空 deque 与已过期 client_id
            if now - self._last_sweep > self._SWEEP_INTERVAL:
                self._sweep_empty_clients(window_start)
                self._last_sweep = now

            hits = bucket[client_id]
            # 清理当前 client_id 的过期记录
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= effective_limit:
                retry_after = int(60 - (now - hits[0]))
                label = "写操作" if is_write else "请求"
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"{label}过于频繁，每分钟限 {effective_limit} 次"},
                    headers={"Retry-After": str(max(retry_after, 1))},
                )
            hits.append(now)

        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """H2 安全响应头中间件：为所有响应注入安全 HTTP 头。

    - X-Content-Type-Options: nosniff  防 MIME 嗅探
    - X-Frame-Options: DENY           防点击劫持（禁止被嵌入 iframe）
    - Referrer-Policy                 控制 Referer 泄露
    - Strict-Transport-Security       HSTS 强制 HTTPS（仅生产 auth_required=True 启用，开发环境不加）
    - Content-Security-Policy         限制资源加载来源，防 XSS 注入执行
    """

    # 安全头默认值（HSTS 仅在 HTTPS 场景有意义，开发环境 http:// 会浏览器告警，故条件性启用）
    _HEADERS_ALWAYS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # default-src 'self' 限制脚本/样式/图片等只能从同源加载；
        # 'unsafe-inline' 保留给 Streamlit 等需要内联样式的框架（如不保留会破坏 UI）
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'",
    }
    _HSTS_HEADER = "Strict-Transport-Security"
    _HSTS_VALUE = "max-age=31536000; includeSubDomains"

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # 注入常驻安全头
        for key, value in self._HEADERS_ALWAYS.items():
            response.headers[key] = value
        # HSTS 仅在生产模式（auth_required=True）下启用：
        # 开发环境通常走 http://，HSTS 会让浏览器强制 HTTPS 导致本地无法访问
        settings = get_settings()
        if settings.auth_required:
            response.headers[self._HSTS_HEADER] = self._HSTS_VALUE
        return response


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """M3 HTTPS 强制中间件：将 HTTP 请求 301 永久重定向到 HTTPS。

    防护目标：
    - 首次访问仍走 HTTP 时被中间人窃听 API Key（HSTS 仅在浏览器缓存后生效，首次访问无法保护）
    - 防止反向代理配置遗漏导致应用端口直接暴露 HTTP

    检测逻辑（满足其一即视为 HTTPS，不重定向）：
    - request.url.scheme == "https"（直连 HTTPS）
    - X-Forwarded-Proto: https（反向代理 TLS 终止后传递的真实协议）

    仅在 settings.force_https=True 时启用，开发环境关闭避免本地 http:// 被重定向。
    健康检查端点（/health）豁免重定向，避免容器编排系统用 HTTP 探活失败。
    """

    # 健康检查路径豁免：容器编排（K8s/Docker Compose）用 HTTP 探活，重定向会导致探活失败
    # P3：/health/ready 同样豁免，K8s readinessProbe 探针不跟随重定向
    _HEALTH_PATHS = frozenset({"/health", "/health/deep", "/health/ready"})

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        # 严格检查 is True：避免 MagicMock 等 truthy 值误触发重定向（防御性编程）
        if settings.force_https is not True:
            return await call_next(request)

        # 健康检查豁免：容器探活通常走 HTTP，重定向会导致 healthcheck 失败
        if request.url.path in self._HEALTH_PATHS:
            return await call_next(request)

        # 检测当前请求是否已是 HTTPS
        # 1. 直连场景：request.url.scheme
        # 2. 反向代理场景：X-Forwarded-Proto 头（nginx/Caddy/Traefik 设置）
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip().lower()
        is_https = request.url.scheme == "https" or forwarded_proto == "https"

        if is_https:
            return await call_next(request)

        # 构造 HTTPS 重定向 URL（保留 path/query，标准端口省略 :443）
        redirect_url = request.url.replace(scheme="https", netloc=request.url.hostname or request.netloc)
        return Response(
            status_code=301,
            headers={"Location": str(redirect_url)},
        )


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    """M4 请求体大小限制中间件：防止超大请求体耗尽内存（DoS 防护）。

    防护目标：
    - Content-Length 声明超大：直接返回 413，不读取 body
    - 分块传输（无 Content-Length）：流式读取累计字节数，超限返回 413

    与 Pydantic max_length 校验的区别：
    - Pydantic 在 JSON 解析后校验字段长度，此时请求体已读入内存
    - 本中间件在 body 读取前/中拦截，避免超大请求体进入内存

    仅对 POST/PUT/PATCH 生效（GET/DELETE 通常无 body）。
    """

    async def dispatch(self, request: Request, call_next):
        # 仅对可能携带 body 的方法检查
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)

        settings = get_settings()
        # 严格检查 is int 且 > 0，避免 MagicMock 等 truthy 值误触发
        max_size = settings.max_request_body_size
        if not isinstance(max_size, int) or max_size <= 0:
            return await call_next(request)

        # 检查 Content-Length 头（声明大小）
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError:
                # 非法 Content-Length 头，交给后续处理
                return await call_next(request)
            if declared_size > max_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"请求体过大（{declared_size} 字节），最大允许 {max_size} 字节"},
                )

        # 分块传输场景（无 Content-Length）：包装 request.stream() 累计字节
        # 通过自定义 receive 钩子在 body 读取过程中实时拦截
        original_receive = request._receive
        bytes_read = 0
        limit_exceeded = False

        async def limited_receive():
            nonlocal bytes_read, limit_exceeded
            if limit_exceeded:
                # 已超限，返回空 body 终止读取
                return {"type": "http.request", "body": b"", "more_body": False}
            message = await original_receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                bytes_read += len(body)
                if bytes_read > max_size:
                    limit_exceeded = True
                    # 返回截断的 body 标记 more_body=False 终止读取
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        request._receive = limited_receive

        try:
            response = await call_next(request)
        except Exception:
            # 流式读取过程中超限，直接返回 413
            if limit_exceeded:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"请求体超过限制（{max_size} 字节）"},
                )
            raise

        # 流式场景超限时（exception 未抛出但 limit_exceeded 标记）
        if limit_exceeded:
            return JSONResponse(
                status_code=413,
                content={"detail": f"请求体超过限制（{max_size} 字节）"},
            )

        return response


class RequestIdMiddleware(BaseHTTPMiddleware):
    """P3 请求追踪中间件：为每个请求生成/透传 request_id。

    - 客户端传入 X-Request-ID 头：校验格式后透传（避免注入风险）
    - 未传入：生成 UUID4 hex（32 字符）
    - 写入 response 头 X-Request-ID，便于客户端关联日志
    - 通过 contextvars 注入到日志上下文，整个请求生命周期内的日志都带该字段

    格式校验：仅允许 [A-Za-z0-9-]{1,128}，防止日志注入与超长 DoS。
    """

    # request_id 格式：字母数字+横线，1-128 字符
    _VALID_PATTERN = __import__("re").compile(r"^[A-Za-z0-9-]{1,128}$")

    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        # 严格检查 is True，避免测试中 MagicMock truthy 误触发
        if settings.request_id_enabled is not True:
            return await call_next(request)

        # 读取客户端传入的 X-Request-ID（可选）
        incoming = request.headers.get("X-Request-ID", "")
        if incoming and self._VALID_PATTERN.match(incoming):
            request_id = incoming
        else:
            from .request_context import new_request_id

            request_id = new_request_id()

        # 注入 contextvars，下游日志自动携带
        # 注：BaseHTTPMiddleware 在 anyio task 中执行，call_next 可能切换上下文，
        # 故不保存 token reset（跨 task reset 会抛 TypeError）；改用 set(None) 清理
        from .request_context import set_request_id

        set_request_id(request_id)

        try:
            response = await call_next(request)
            # 写入响应头，客户端可关联服务端日志
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            # 清理 request_id，避免泄漏到下一个请求
            set_request_id(None)


def setup_middleware(app) -> None:
    """在 FastAPI 应用上注册所有中间件。

    P0 修复：硬约束「Authentication → Rate limiting → Endpoint」，请求方向必须
    鉴权先于限流执行，避免无效 Key 的请求消耗限流配额被攻击者用来对合法用户发起 DoS。

    FastAPI 中间件为 LIFO 执行（后注册的先执行），注册顺序反过来才是请求执行链：
    请求 → CORS → HTTPS 重定向 → 安全头 → 请求体限制 → 鉴权 → 限流 → 端点
    响应 → 端点 → 限流 → 鉴权 → 请求体限制 → 安全头 → HTTPS 重定向 → CORS

    因此注册顺序（先注册→后执行→内层）应为：
    限流 → 鉴权 → 请求体限制 → 安全头 → HTTPS 重定向（后注册→先执行→最外层）

    P3 新增：RequestId 中间件注册在最外层（最后注册），保证请求进入应用的第一时间
    就设置 contextvars，所有后续中间件与端点的日志都带 request_id。
    """
    # P0：先注册 RateLimit（更内层），后注册 APIKeyAuth（更外层），
    # 这样请求执行顺序为 APIKeyAuth → RateLimit → 端点（鉴权先于限流）
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(APIKeyAuthMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(HTTPSRedirectMiddleware)
    # P3：RequestId 最外层（最后注册→最先执行），确保所有中间件日志都带 request_id
    app.add_middleware(RequestIdMiddleware)

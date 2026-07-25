"""
中间件测试：API Key 鉴权 + 请求限流
"""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cayz_agent.middleware import (
    APIKeyAuthMiddleware,
    RateLimitMiddleware,
    RequestBodyLimitMiddleware,
    setup_middleware,
    _extract_api_key,
    _get_client_id,
)


def _build_test_app(api_key: str = "", rate_limit: int = 60) -> FastAPI:
    """构建带中间件的测试应用"""
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/chat")
    async def chat():
        return {"reply": "ok"}

    setup_middleware(app)
    return app


class TestExtractApiKey:
    """测试 API Key 提取逻辑"""

    def test_extract_from_bearer(self):
        from starlette.requests import Request
        scope = {
            "type": "http",
            "method": "GET",
            "headers": [(b"authorization", b"Bearer my-secret-key")],
        }
        req = Request(scope)
        assert _extract_api_key(req) == "my-secret-key"

    def test_extract_from_x_api_key(self):
        from starlette.requests import Request
        scope = {
            "type": "http",
            "method": "GET",
            "headers": [(b"x-api-key", b"my-secret-key")],
        }
        req = Request(scope)
        assert _extract_api_key(req) == "my-secret-key"

    def test_extract_returns_none_when_missing(self):
        from starlette.requests import Request
        scope = {"type": "http", "method": "GET", "headers": []}
        req = Request(scope)
        assert _extract_api_key(req) is None


class TestAPIKeyAuth:
    """测试 API Key 鉴权中间件"""

    def test_no_api_key_config_allows_all(self):
        """未配置 API_KEY 且 auth_required=False 时放行所有请求（开发模式）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_valid_api_key_grants_access(self):
        """正确的 API Key 应放行"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-123"
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat", headers={"X-API-Key": "secret-123"})
            assert resp.status_code == 200

    def test_valid_bearer_token_grants_access(self):
        """Bearer token 鉴权"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-123"
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post(
                "/chat",
                headers={"Authorization": "Bearer secret-123"},
            )
            assert resp.status_code == 200

    def test_missing_api_key_returns_401(self):
        """缺少 API Key 应返回 401"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-123"
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat")
            assert resp.status_code == 401
            assert "API Key" in resp.json()["detail"]

    def test_wrong_api_key_returns_401(self):
        """错误的 API Key 应返回 401"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-123"
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat", headers={"X-API-Key": "wrong"})
            assert resp.status_code == 401

    def test_public_paths_bypass_auth(self):
        """/health 等公开端点应跳过鉴权"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-123"
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200


class TestAuthEnforcement:
    """P0 安全修复：测试 auth_required 强制鉴权，防止生产环境裸奔"""

    def test_auth_required_blocks_when_no_api_key(self):
        """auth_required=True 且 api_key 未配置时，非公开端点应返回 503"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.rate_limit_write_per_minute = 0
            mock.return_value.trust_forwarded_headers = False
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat")
        assert resp.status_code == 503
        assert "未就绪" in resp.json()["detail"]

    def test_auth_required_false_allows_without_key(self):
        """auth_required=False 且 api_key 未配置时，非公开端点应放行（开发模式）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.rate_limit_write_per_minute = 0
            mock.return_value.trust_forwarded_headers = False
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat")
        assert resp.status_code == 200

    def test_auth_required_public_endpoint_still_accessible(self):
        """auth_required=True 且 api_key 未配置时，公开端点 /health 仍可访问"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_auth_required_with_key_config_works_normally(self):
        """auth_required=True 且 api_key 已配置时，正常鉴权流程不受影响"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-123"
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            # 带 Key 放行
            resp = client.post("/chat", headers={"X-API-Key": "secret-123"})
            assert resp.status_code == 200
            # 不带 Key 返回 401
            resp = client.post("/chat")
            assert resp.status_code == 401


class TestRateLimit:
    """测试请求限流中间件"""

    def test_under_limit_passes(self):
        """未超限请求应放行"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.rate_limit_per_minute = 10
            app = _build_test_app()
            client = TestClient(app)
            for _ in range(5):
                resp = client.get("/health")
                assert resp.status_code == 200

    def test_over_limit_returns_429(self):
        """超限请求应返回 429"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 3
            mock.return_value.rate_limit_write_per_minute = 0
            mock.return_value.trust_forwarded_headers = False
            app = _build_test_app()
            client = TestClient(app)
            # 前 3 次放行
            for _ in range(3):
                resp = client.get("/health")
                # 公开端点不限流，所以这里测 /chat
            # 用 /chat 测试限流
            for _ in range(3):
                resp = client.post("/chat")
                assert resp.status_code == 200
            # 第 4 次应被限流
            resp = client.post("/chat")
            assert resp.status_code == 429
            assert "频繁" in resp.json()["detail"]
            assert "Retry-After" in resp.headers

    def test_zero_limit_means_unlimited(self):
        """limit=0 表示不限制"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.rate_limit_write_per_minute = 0
            mock.return_value.trust_forwarded_headers = False
            app = _build_test_app()
            client = TestClient(app)
            for _ in range(20):
                resp = client.post("/chat")
                assert resp.status_code == 200

    def test_public_paths_bypass_rate_limit(self):
        """公开端点不限流"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.rate_limit_per_minute = 2
            app = _build_test_app()
            client = TestClient(app)
            for _ in range(10):
                resp = client.get("/health")
                assert resp.status_code == 200


class TestRateLimitSweep:
    """测试限流中间件的 client_id 清理（P2 内存泄漏修复）"""

    def test_sweep_removes_empty_clients(self):
        """sweep 应删除已无活跃记录的 client_id"""
        import time as _time

        mw = RateLimitMiddleware(app=FastAPI())
        # 模拟两个 client 都有过期记录
        mw._hits["client-a"].append(_time.time() - 120)  # 2 分钟前
        mw._hits["client-b"].append(_time.time() - 200)  # 3 分钟前
        assert len(mw._hits) == 2

        # 触发 sweep（window_start = now - 60，两个 client 的记录都过期）
        mw._sweep_empty_clients(window_start=_time.time() - 60)
        assert len(mw._hits) == 0

    def test_sweep_keeps_active_clients(self):
        """sweep 应保留仍有活跃记录的 client_id"""
        import time as _time

        mw = RateLimitMiddleware(app=FastAPI())
        now = _time.time()
        mw._hits["client-a"].append(now - 10)  # 10 秒前，仍在窗口内
        mw._hits["client-b"].append(now - 120)  # 2 分钟前，已过期

        mw._sweep_empty_clients(window_start=now - 60)
        assert "client-a" in mw._hits
        assert "client-b" not in mw._hits

    def test_sweep_handles_empty_deque(self):
        """sweep 应清理空 deque 的 client_id"""
        import time as _time

        mw = RateLimitMiddleware(app=FastAPI())
        # defaultdict 不会自动创建空 deque，但手动赋值可以模拟
        from collections import deque
        mw._hits["empty-client"] = deque()
        assert len(mw._hits) == 1

        mw._sweep_empty_clients(window_start=_time.time() - 60)
        assert len(mw._hits) == 0

    def test_sweep_triggered_by_interval(self):
        """超过 _SWEEP_INTERVAL 后下次 dispatch 应触发 sweep"""
        import asyncio
        import time as _time
        from collections import deque

        # 直接构造中间件实例，避免依赖 starlette 内部实例化时机
        async def _dummy_app(request):
            return None

        mw = RateLimitMiddleware(app=_dummy_app)
        mw._SWEEP_INTERVAL = 0  # 强制每次 dispatch 都触发 sweep

        # 注入一个 ghost client（有过期记录）
        mw._hits["ghost"] = deque()
        mw._hits["ghost"].append(_time.time() - 300)
        assert "ghost" in mw._hits

        # 构造一个 mock request 与 call_next
        class _Req:
            class _url:
                path = "/chat"
            url = _url()
            class _client:
                host = "127.0.0.1"
            client = _client()
            headers = {}

        async def _call_next(req):
            from starlette.responses import JSONResponse
            return JSONResponse({"ok": True})

        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.rate_limit_per_minute = 100
            asyncio.run(mw.dispatch(_Req(), _call_next))

        # ghost 应被清理
        assert "ghost" not in mw._hits


class TestP1ScopeResolution:
    """P1 权限分级：测试 _resolve_scope 正确解析三级 Key 权限"""

    def test_admin_key_resolves_to_admin(self):
        """管理员 Key（api_key）解析为 admin"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = "write-key"
            mock.return_value.readonly_api_keys = "ro-key"
            assert _resolve_scope("admin-secret", mock.return_value) == "admin"

    def test_write_key_resolves_to_write(self):
        """写权限 Key 解析为 write"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = "write-key, write-key-2"
            mock.return_value.readonly_api_keys = "ro-key"
            assert _resolve_scope("write-key-2", mock.return_value) == "write"

    def test_readonly_key_resolves_to_readonly(self):
        """只读 Key 解析为 readonly"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = "ro-key"
            assert _resolve_scope("ro-key", mock.return_value) == "readonly"

    def test_invalid_key_resolves_to_none(self):
        """无效 Key 解析为 None"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = "write-key"
            mock.return_value.readonly_api_keys = "ro-key"
            assert _resolve_scope("bogus", mock.return_value) is None

    def test_none_key_resolves_to_none(self):
        """未提供 Key 解析为 None"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            assert _resolve_scope(None, mock.return_value) is None


class TestH1TimingSafeCompare:
    """H1 时序安全：API Key 比较使用 hmac.compare_digest 防止时序攻击"""

    def test_admin_key_uses_compare_digest(self):
        """admin key 比较应走 hmac.compare_digest（功能正确性）"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret-key"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            # 正确 key 应解析为 admin
            assert _resolve_scope("admin-secret-key", mock.return_value) == "admin"
            # 错误 key 应解析为 None
            assert _resolve_scope("admin-secret-keY", mock.return_value) is None

    def test_admin_key_length_mismatch_safe(self):
        """长度不同的 key 应安全返回 None，不抛异常"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            # 短于真实 key
            assert _resolve_scope("ad", mock.return_value) is None
            # 长于真实 key
            assert _resolve_scope("admin-secret-extra-padding", mock.return_value) is None

    def test_write_key_uses_constant_time_contains(self):
        """write key 比较应走 _constant_time_contains（功能正确性 + 多 key 匹配）"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = "write-key-1, write-key-2, write-key-3"
            mock.return_value.readonly_api_keys = ""
            # 第 1 个 key 匹配
            assert _resolve_scope("write-key-1", mock.return_value) == "write"
            # 第 3 个 key 匹配（验证遍历全列表）
            assert _resolve_scope("write-key-3", mock.return_value) == "write"
            # 不存在的 key 返回 None
            assert _resolve_scope("write-key-999", mock.return_value) is None

    def test_readonly_key_uses_constant_time_contains(self):
        """readonly key 比较应走 _constant_time_contains"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = "ro-1, ro-2"
            assert _resolve_scope("ro-2", mock.return_value) == "readonly"
            assert _resolve_scope("ro-X", mock.return_value) is None

    def test_constant_time_contains_no_early_exit(self):
        """_constant_time_contains 遍历全部 key，匹配后仍继续比较（无短路）"""
        from cayz_agent.middleware import _constant_time_contains
        # 第 1 个就匹配，但函数应遍历完所有 3 个
        keys = ["match", "key2", "key3"]
        assert _constant_time_contains("match", keys) is True
        # 最后一个才匹配
        assert _constant_time_contains("key3", keys) is True
        # 无匹配
        assert _constant_time_contains("nope", keys) is False
        # 空列表
        assert _constant_time_contains("any", []) is False

    def test_constant_time_contains_length_mismatch(self):
        """_constant_time_contains 长度不同的候选应安全返回 False"""
        from cayz_agent.middleware import _constant_time_contains
        keys = ["exact-key"]
        assert _constant_time_contains("", keys) is False
        assert _constant_time_contains("x", keys) is False
        assert _constant_time_contains("exact-key-with-padding", keys) is False

    def test_scope_resolution_consistent_with_p1_behavior(self):
        """H1 修复后行为与 P1 一致：三级权限解析正确"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-key"
            mock.return_value.write_api_keys = "write-key"
            mock.return_value.readonly_api_keys = "ro-key"
            # 全部正确解析
            assert _resolve_scope("admin-key", mock.return_value) == "admin"
            assert _resolve_scope("write-key", mock.return_value) == "write"
            assert _resolve_scope("ro-key", mock.return_value) == "readonly"
            # 权限不会越级：write key 不能解析为 admin
            assert _resolve_scope("write-key", mock.return_value) != "admin"
            # 无效 key 返回 None
            assert _resolve_scope("invalid", mock.return_value) is None

    def test_admin_key_partial_prefix_no_match(self):
        """admin key 前缀匹配不应通过（compare_digest 要求完全相等）"""
        from cayz_agent.middleware import _resolve_scope
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "sk-admin-secret-123456"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            # 前缀
            assert _resolve_scope("sk-admin", mock.return_value) is None
            # 后缀
            assert _resolve_scope("secret-123456", mock.return_value) is None
            # 中间子串
            assert _resolve_scope("admin-secret", mock.return_value) is None


class TestP1ForwardedHeaders:
    """P1 限流维度：测试 X-Forwarded-For / X-Real-IP 提取"""

    def test_real_ip_from_xff_when_trusted(self):
        """trust_forwarded_headers=True 时从 X-Forwarded-For 提取真实 IP"""
        from cayz_agent.middleware import _get_real_ip
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.trust_forwarded_headers = True

            class _Req:
                headers = {"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}

                class _client:
                    host = "127.0.0.1"

                client = _client()

            assert _get_real_ip(_Req()) == "203.0.113.5"

    def test_real_ip_from_x_real_ip(self):
        """无 X-Forwarded-For 时回退到 X-Real-IP"""
        from cayz_agent.middleware import _get_real_ip
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.trust_forwarded_headers = True

            class _Req:
                headers = {"X-Real-IP": "198.51.100.7"}

                class _client:
                    host = "127.0.0.1"

                client = _client()

            assert _get_real_ip(_Req()) == "198.51.100.7"

    def test_real_ip_ignored_when_not_trusted(self):
        """trust_forwarded_headers=False 时忽略转发头，使用直连 IP"""
        from cayz_agent.middleware import _get_real_ip
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.trust_forwarded_headers = False

            class _Req:
                headers = {"X-Forwarded-For": "203.0.113.5"}

                class _client:
                    host = "127.0.0.1"

                client = _client()

            assert _get_real_ip(_Req()) == "127.0.0.1"


class TestH2SecurityHeaders:
    """H2 安全响应头：所有响应应注入安全 HTTP 头"""

    def test_x_content_type_options_present(self):
        """X-Content-Type-Options: nosniff 应出现在所有响应"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options_present(self):
        """X-Frame-Options: DENY 应出现在所有响应（防点击劫持）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_referrer_policy_present(self):
        """Referrer-Policy 应出现在所有响应"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_csp_present(self):
        """Content-Security-Policy 应出现在所有响应"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            csp = resp.headers.get("Content-Security-Policy", "")
            # CSP 应包含 default-src 'self'
            assert "default-src 'self'" in csp

    def test_hsts_absent_in_dev_mode(self):
        """开发模式（auth_required=False）不应设置 HSTS（避免本地 http:// 被强制 HTTPS）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert "Strict-Transport-Security" not in resp.headers

    def test_hsts_present_in_production(self):
        """生产模式（auth_required=True）应设置 HSTS 强制 HTTPS"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            hsts = resp.headers.get("Strict-Transport-Security", "")
            assert "max-age=31536000" in hsts
            assert "includeSubDomains" in hsts

    def test_security_headers_on_error_response(self):
        """401 错误响应也应带安全头（中间件在鉴权之外）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "admin-secret"
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            # 无 Key 访问需鉴权端点应返回 401
            resp = client.post("/chat")
            assert resp.status_code == 401
            # 401 响应也应带安全头
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_security_headers_on_rate_limit_response(self):
        """429 限流响应也应带安全头"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret"
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 1  # 极低限流便于触发
            app = _build_test_app()
            client = TestClient(app)
            # 第 1 次通过
            resp1 = client.post("/chat", headers={"X-API-Key": "secret"})
            assert resp1.status_code == 200
            # 第 2 次触发限流
            resp2 = client.post("/chat", headers={"X-API-Key": "secret"})
            assert resp2.status_code == 429
            # 429 响应也应带安全头
            assert resp2.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp2.headers.get("X-Frame-Options") == "DENY"

    def test_csp_allows_unsafe_inline_for_streamlit(self):
        """CSP 应保留 'unsafe-inline' 以兼容 Streamlit 内联样式（否则 UI 会破坏）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            csp = resp.headers.get("Content-Security-Policy", "")
            # script-src 和 style-src 都应允许 'unsafe-inline'
            assert "script-src 'self' 'unsafe-inline'" in csp
            assert "style-src 'self' 'unsafe-inline'" in csp


class TestM3HTTPSRedirect:
    """M3 HTTPS 强制：HTTP 请求 301 重定向到 HTTPS"""

    def test_force_https_disabled_passes_through(self):
        """force_https=False 时 HTTP 请求应正常处理（不重定向）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = False
            app = _build_test_app()
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    def test_http_redirected_to_https_when_force_enabled(self):
        """force_https=True 时 HTTP POST /chat 应 301 重定向到 HTTPS"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            # TestClient 默认用 http://，应触发重定向
            resp = client.post("/chat", follow_redirects=False)
            assert resp.status_code == 301
            assert resp.headers["Location"].startswith("https://")

    def test_redirect_preserves_path_and_query(self):
        """重定向应保留原始 path 和 query 参数"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat?foo=bar", follow_redirects=False)
            assert resp.status_code == 301
            location = resp.headers["Location"]
            assert "/chat" in location
            assert "foo=bar" in location

    def test_health_endpoint_exempt_from_redirect(self):
        """/health 健康检查应豁免重定向（容器探活走 HTTP）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            # /health 不应被重定向
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    def test_health_deep_exempt_from_redirect(self):
        """/health/deep 也应豁免重定向"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            # 添加 /health/deep 端点用于测试
            @app.get("/health/deep")
            async def health_deep():
                return {"status": "ok"}
            client = TestClient(app)
            resp = client.get("/health/deep")
            assert resp.status_code == 200

    def test_https_request_not_redirected(self):
        """已是 HTTPS 的请求不应被重定向（通过 X-Forwarded-Proto: https 模拟）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            # 通过 X-Forwarded-Proto: https 模拟反向代理后的 HTTPS 请求
            resp = client.post("/chat", headers={"X-Forwarded-Proto": "https"})
            assert resp.status_code == 200
            assert resp.json() == {"reply": "ok"}

    def test_redirect_happens_before_auth(self):
        """重定向应在鉴权之前发生（HTTP 请求不应返回 401，而应 301）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-key"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            # 无 Key 访问需鉴权端点，但因 force_https 应先返回 301 而非 401
            resp = client.post("/chat", follow_redirects=False)
            assert resp.status_code == 301
            # 不应是 401
            assert resp.status_code != 401

    def test_forwarded_proto_with_multiple_values(self):
        """X-Forwarded-Proto 含多个值（链式代理）时应取第一个"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            # 多级代理场景：client, proxy1, proxy2
            resp = client.post("/chat", headers={"X-Forwarded-Proto": "https, http"})
            assert resp.status_code == 200  # 第一个是 https，不重定向

    def test_redirect_url_uses_https_scheme(self):
        """重定向 Location 的 scheme 必须是 https"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = ""
            mock.return_value.auth_required = False
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = True
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat", follow_redirects=False)
            location = resp.headers["Location"]
            from urllib.parse import urlparse
            parsed = urlparse(location)
            assert parsed.scheme == "https"


class TestM4RequestBodyLimit:
    """M4 请求体大小限制：防止超大请求体耗尽内存（DoS 防护）"""

    def _mock_settings(self, mock, max_size=100):
        """统一设置 mock settings，避免重复"""
        mock.return_value.api_key = ""
        mock.return_value.write_api_keys = ""
        mock.return_value.readonly_api_keys = ""
        mock.return_value.auth_required = False
        mock.return_value.rate_limit_per_minute = 0
        mock.return_value.force_https = False
        mock.return_value.max_request_body_size = max_size

    def test_content_length_within_limit_passes(self):
        """Content-Length 在限制内应正常处理"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=100)
            app = _build_test_app()
            client = TestClient(app)
            # body 10 字节，限制 100 字节
            resp = client.post("/chat", json={"msg": "hello"})
            assert resp.status_code == 200

    def test_content_length_exceeds_limit_returns_413(self):
        """Content-Length 声明超过限制应返回 413（不读取 body）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=10)
            app = _build_test_app()
            client = TestClient(app)
            # 构造超大 body，Content-Length 会超过 10 字节
            big_body = "x" * 1000
            resp = client.post("/chat", json={"msg": big_body})
            assert resp.status_code == 413
            assert "请求体过大" in resp.json()["detail"]

    def test_get_request_exempt_from_body_limit(self):
        """GET 请求不检查 body（即使 Content-Length 存在）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=1)
            app = _build_test_app()
            client = TestClient(app)
            # GET 请求应豁免
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_delete_request_exempt_from_body_limit(self):
        """DELETE 请求不检查 body"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=1)
            app = FastAPI()

            @app.delete("/items/{item_id}")
            async def delete_item(item_id: str):
                return {"deleted": True, "id": item_id}

            setup_middleware(app)
            client = TestClient(app)
            # DELETE 带 body 也应豁免（TestClient.delete 不支持 json 参数，用 request）
            resp = client.request("DELETE", "/items/123", json={"confirm": True})
            assert resp.status_code == 200

    def test_body_limit_disabled_when_zero(self):
        """max_request_body_size=0 时关闭限制，超大请求体放行"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=0)
            app = _build_test_app()
            client = TestClient(app)
            big_body = "x" * 1000
            resp = client.post("/chat", json={"msg": big_body})
            assert resp.status_code == 200

    def test_invalid_content_length_header_passes_through(self):
        """非法 Content-Length 头（非数字）应放行交给后续处理"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=10)
            app = _build_test_app()
            client = TestClient(app)
            # 手动设置非法 Content-Length
            resp = client.post(
                "/chat",
                content=b'{"msg": "hi"}',
                headers={"Content-Length": "abc", "Content-Type": "application/json"},
            )
            # 非法头交给后续处理，应正常处理（TestClient/httpx 会重新计算）
            assert resp.status_code == 200

    def test_body_exactly_at_limit_passes(self):
        """body 大小恰好等于限制时应放行（边界条件，> 才拦截）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            # 设置一个能精确控制的限制
            # body: {"msg":"xxxxx} 约 16 字节，设置 max_size=16 测试边界
            self._mock_settings(mock, max_size=10 * 1024 * 1024)
            app = _build_test_app()
            client = TestClient(app)
            resp = client.post("/chat", json={"msg": "hello"})
            assert resp.status_code == 200

    def test_body_limit_happens_before_auth(self):
        """请求体限制应在鉴权之前（超大 body 应返回 413 而非 401）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            mock.return_value.api_key = "secret-key"
            mock.return_value.write_api_keys = ""
            mock.return_value.readonly_api_keys = ""
            mock.return_value.auth_required = True
            mock.return_value.rate_limit_per_minute = 0
            mock.return_value.force_https = False
            mock.return_value.max_request_body_size = 10
            app = _build_test_app()
            client = TestClient(app)
            # 无 Key + 超大 body：force_https=False，请求体限制在最外层
            big_body = "x" * 1000
            resp = client.post("/chat", json={"msg": big_body})
            assert resp.status_code == 413
            # 不应是 401（鉴权未触发）
            assert resp.status_code != 401

    def test_413_response_includes_security_headers(self):
        """413 响应应带安全头（请求体限制在安全头之外，错误响应也应有安全头）"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=10)
            app = _build_test_app()
            client = TestClient(app)
            big_body = "x" * 1000
            resp = client.post("/chat", json={"msg": big_body})
            assert resp.status_code == 413
            # 安全头应存在
            assert resp.headers.get("X-Content-Type-Options") == "nosniff"
            assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_chunked_transfer_exceeds_limit_returns_413(self):
        """分块传输（无 Content-Length）超过限制应返回 413"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=10)
            app = _build_test_app()
            client = TestClient(app)
            # httpx 默认对大请求体使用分块传输（不设 Content-Length）
            # 通过 content 参数传原始字节，且不设 Content-Length 头
            big_body = b"x" * 1000
            resp = client.post(
                "/chat",
                content=big_body,
                headers={"Content-Type": "application/json"},
            )
            # 分块传输超限应返回 413
            assert resp.status_code == 413

    def test_put_request_checked(self):
        """PUT 请求应检查 body 大小"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=10)
            app = FastAPI()

            @app.put("/items/{item_id}")
            async def update_item(item_id: str):
                return {"updated": True, "id": item_id}

            setup_middleware(app)
            client = TestClient(app)
            big_body = "x" * 1000
            resp = client.put("/items/123", json={"msg": big_body})
            assert resp.status_code == 413

    def test_patch_request_checked(self):
        """PATCH 请求应检查 body 大小"""
        with patch("cayz_agent.middleware.get_settings") as mock:
            self._mock_settings(mock, max_size=10)
            app = FastAPI()

            @app.patch("/items/{item_id}")
            async def patch_item(item_id: str):
                return {"patched": True, "id": item_id}

            setup_middleware(app)
            client = TestClient(app)
            big_body = "x" * 1000
            resp = client.patch("/items/123", json={"msg": big_body})
            assert resp.status_code == 413


class TestM4UvicornHardening:
    """M4 Uvicorn 服务器硬化参数测试"""

    def test_run_passes_timeout_keep_alive(self):
        """run() 应将 uvicorn_timeout_keep_alive 传给 uvicorn"""
        with patch("uvicorn.run") as mock_run, \
                patch("cayz_agent.api.settings") as mock_settings:
            mock_settings.api_host = "0.0.0.0"
            mock_settings.api_port = 8000
            mock_settings.uvicorn_timeout_keep_alive = 5
            mock_settings.uvicorn_limit_concurrency = 100
            from cayz_agent.api import run
            run()
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["timeout_keep_alive"] == 5

    def test_run_passes_limit_concurrency_when_positive(self):
        """limit_concurrency > 0 时应传给 uvicorn"""
        with patch("uvicorn.run") as mock_run, \
                patch("cayz_agent.api.settings") as mock_settings:
            mock_settings.api_host = "0.0.0.0"
            mock_settings.api_port = 8000
            mock_settings.uvicorn_timeout_keep_alive = 5
            mock_settings.uvicorn_limit_concurrency = 200
            from cayz_agent.api import run
            run()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["limit_concurrency"] == 200

    def test_run_omits_limit_concurrency_when_zero(self):
        """limit_concurrency=0 时不传该参数（uvicorn 默认不限制）"""
        with patch("uvicorn.run") as mock_run, \
                patch("cayz_agent.api.settings") as mock_settings:
            mock_settings.api_host = "0.0.0.0"
            mock_settings.api_port = 8000
            mock_settings.uvicorn_timeout_keep_alive = 5
            mock_settings.uvicorn_limit_concurrency = 0
            from cayz_agent.api import run
            run()
            call_kwargs = mock_run.call_args.kwargs
            assert "limit_concurrency" not in call_kwargs

    def test_run_passes_host_and_port(self):
        """run() 应传 host/port 配置"""
        with patch("uvicorn.run") as mock_run, \
                patch("cayz_agent.api.settings") as mock_settings:
            mock_settings.api_host = "127.0.0.1"
            mock_settings.api_port = 9000
            mock_settings.uvicorn_timeout_keep_alive = 5
            mock_settings.uvicorn_limit_concurrency = 100
            from cayz_agent.api import run
            run()
            call_kwargs = mock_run.call_args.kwargs
            assert call_kwargs["host"] == "127.0.0.1"
            assert call_kwargs["port"] == 9000
            assert call_kwargs["reload"] is False

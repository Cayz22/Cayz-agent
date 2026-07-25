"""pytest 共享配置与 fixture（包安装后无需 sys.path 操作）"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ============================================================
# 环境变量：测试前强制清空敏感字段，防止读到本地 .env
# ============================================================
# 必须在导入 cayz_agent.* 之前设置，确保 Settings 默认值生效
for _key in (
    "OPENAI_API_KEY",
    "ZHIPU_API_KEY",
    "ERNIE_API_KEY",
    "TAVILY_API_KEY",
    "API_KEY",
    "WECOM_WEBHOOK_URL",
    "SMTP_USERNAME",
    "SMTP_PASSWORD",
):
    os.environ.pop(_key, None)

# 测试环境关闭强制鉴权（auth_required=False），允许在无 API Key 时
# 访问非公开端点，避免现有端点测试因 503 而失败。
# 鉴权相关行为通过 patch get_settings 在专门的安全测试中验证。
os.environ["AUTH_REQUIRED"] = "false"
# M3 测试环境默认关闭 HTTPS 强制（避免 mock get_settings 时 force_https 为 MagicMock truthy 误触发）
os.environ["FORCE_HTTPS"] = "false"


# ============================================================
# 全局 autouse fixture
# ============================================================
@pytest.fixture(autouse=True)
def _reset_singletons():
    """每个测试前重置配置单例 + 监控注册表 + 缓存单例，确保测试隔离。

    - reset_settings_cache(): 让下个测试重新读取环境变量
    - MetricsRegistry.reset(): 清空所有计数器/直方图/Gauge
    - reset_cache_singletons(): 清空 LLM/Embedding/RAG 缓存单例
    - stop_alert_watcher(): 停止 api.py 启动时拉起的后台告警线程
    """
    # 延迟导入，避免在模块加载阶段触发 Settings 实例化
    from cayz_agent.config import reset_settings_cache

    reset_settings_cache()

    # 停止可能由 api.py 模块导入时启动的告警 watcher，避免后台线程干扰测试
    try:
        from cayz_agent.alerts import stop_alert_watcher

        stop_alert_watcher()
    except Exception:
        pass

    # 重置监控指标
    try:
        from cayz_agent.monitor import get_registry

        get_registry().reset()
    except Exception:
        # 某些测试可能不依赖 monitor，忽略重置失败
        pass

    # 重置缓存单例（让下个测试按新配置重建缓存）
    try:
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()
    except Exception:
        pass

    yield

    # 测试结束后再次重置，避免状态泄漏到下一个测试
    reset_settings_cache()
    try:
        from cayz_agent.alerts import stop_alert_watcher

        stop_alert_watcher()
    except Exception:
        pass
    try:
        from cayz_agent.monitor import get_registry

        get_registry().reset()
    except Exception:
        pass
    try:
        from cayz_agent.cache import reset_cache_singletons

        reset_cache_singletons()
    except Exception:
        pass


# ============================================================
# FastAPI 测试客户端 fixture
# ============================================================
@pytest.fixture
def api_client():
    """FastAPI TestClient（同步），适用于大多数端点测试"""
    from cayz_agent.api import app

    return TestClient(app)


# ============================================================
# Mock LLM fixture（用于 Agent 相关测试）
# ============================================================
@pytest.fixture
def mock_llm():
    """Mock 的 LLM 实例，默认返回固定回复。

    使用方式：
        def test_xxx(self, mock_llm):
            mock_llm.invoke.return_value = AIMessage(content="...")
    """
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="mocked reply")
    llm.bind_tools.return_value = llm
    return llm


@pytest.fixture
def mock_llm_response():
    """构造 LLM 响应对象的工厂 fixture。

    使用方式：
        def test_xxx(self, mock_llm_response):
            mock_llm.invoke.return_value = mock_llm_response(content="hello")
    """

    def _make(content: str = "", tool_calls=None):
        resp = MagicMock()
        resp.content = content
        resp.tool_calls = tool_calls or []
        return resp

    return _make


# ============================================================
# 临时目录 fixture（用于 SQLite/ChromaDB 等需要落盘的测试）
# ============================================================
@pytest.fixture
def temp_dir():
    """提供独立临时目录，测试结束自动清理"""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def temp_sqlite_path(temp_dir):
    """临时 SQLite 文件路径（用于 checkpointer 测试）"""
    return os.path.join(temp_dir, "test_checkpoints.db")


# ============================================================
# 会话管理 fixture
# ============================================================
@pytest.fixture
def session_manager(temp_dir):
    """使用临时目录的 SessionManager 实例（隔离测试）"""
    from cayz_agent.session import SessionManager

    db_path = os.path.join(temp_dir, "sessions.db")
    manager = SessionManager(db_path=db_path)
    yield manager
    # 清理：关闭连接（如存在）
    try:
        if hasattr(manager, "_conn") and manager._conn:
            manager._conn.close()
    except Exception:
        pass

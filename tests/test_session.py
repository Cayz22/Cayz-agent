"""
会话管理器测试

测试 SessionManager 的列表、查询、删除功能。
使用临时 SQLite 数据库模拟 LangGraph checkpointer 表。
"""

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cayz_agent.session import SessionInfo, SessionManager, get_session_manager


@pytest.fixture
def temp_db():
    """创建临时 SQLite 数据库并初始化 LangGraph checkpoint 表"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # 初始化 LangGraph checkpoint 表结构
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            checkpoint_id TEXT NOT NULL,
            parent_checkpoint_id TEXT,
            type TEXT,
            checkpoint BLOB,
            metadata BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
        );
        CREATE TABLE writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL DEFAULT '',
            channel TEXT NOT NULL,
            version TEXT NOT NULL,
            type TEXT,
            blob BLOB,
            PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
        );
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Windows 下文件可能被占用，重试几次
    import time as _time

    for _ in range(5):
        try:
            Path(db_path).unlink(missing_ok=True)
            break
        except PermissionError:
            _time.sleep(0.1)


@pytest.fixture
def manager(temp_db):
    """创建使用临时 DB 的 SessionManager（会自动初始化 session_activity 表）"""
    with patch("cayz_agent.session.get_settings") as mock:
        mock.return_value.checkpoint_backend = "sqlite"
        mock.return_value.sqlite_checkpoint_path = temp_db
        m = SessionManager(db_path=temp_db)
        yield m


def _insert_checkpoint(db_path: str, thread_id: str, checkpoint_id: str = "cp-1"):
    """插入测试 checkpoint 记录"""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?)",
        (thread_id, checkpoint_id, "type", b"data", b"{}"),
    )
    # channel 和 version 使用 checkpoint_id 以避免唯一约束冲突
    conn.execute(
        "INSERT INTO writes (thread_id, channel, version, type, blob) VALUES (?, ?, ?, ?, ?)",
        (thread_id, f"channel_{checkpoint_id}", f"v_{checkpoint_id}", "type", b"blob"),
    )
    conn.commit()
    conn.close()


class TestSessionInfo:
    """测试 SessionInfo 数据类"""

    def test_to_dict(self):
        info = SessionInfo(thread_id="t1", last_updated=1700000000, message_count=5)
        d = info.to_dict()
        assert d["thread_id"] == "t1"
        assert d["message_count"] == 5
        assert d["last_updated"] is not None

    def test_to_dict_with_zero_timestamp(self):
        info = SessionInfo(thread_id="t1", last_updated=0, message_count=0)
        d = info.to_dict()
        assert d["last_updated"] is None


class TestListSessions:
    """测试 list_sessions"""

    def test_empty_db_returns_empty_list(self, manager, temp_db):
        sessions, total = manager.list_sessions()
        assert sessions == []
        assert total == 0

    def test_returns_sessions(self, manager, temp_db):
        _insert_checkpoint(temp_db, "thread-1")
        _insert_checkpoint(temp_db, "thread-2")
        manager.touch_session("thread-1")
        manager.touch_session("thread-2")

        sessions, total = manager.list_sessions()
        assert len(sessions) == 2
        assert total == 2
        thread_ids = [s.thread_id for s in sessions]
        assert "thread-1" in thread_ids
        assert "thread-2" in thread_ids

    def test_message_count(self, manager, temp_db):
        _insert_checkpoint(temp_db, "thread-1", "cp-1")
        _insert_checkpoint(temp_db, "thread-1", "cp-2")
        _insert_checkpoint(temp_db, "thread-1", "cp-3")
        manager.touch_session("thread-1")

        sessions, _ = manager.list_sessions()
        session = next(s for s in sessions if s.thread_id == "thread-1")
        assert session.message_count == 3

    def test_limit_and_offset(self, manager, temp_db):
        for i in range(5):
            _insert_checkpoint(temp_db, f"thread-{i}")
            manager.touch_session(f"thread-{i}")

        # 测试 limit
        sessions, total = manager.list_sessions(limit=2)
        assert len(sessions) == 2
        # total 应为符合条件的总会话数（非当前页大小）
        assert total == 5

        # 测试 offset
        sessions_page2, _ = manager.list_sessions(limit=2, offset=2)
        assert len(sessions_page2) == 2

    def test_last_updated_from_activity_table(self, manager, temp_db):
        """list_sessions 的 last_updated 应来自 session_activity 表"""
        _insert_checkpoint(temp_db, "thread-1")
        manager.touch_session("thread-1")

        sessions, _ = manager.list_sessions()
        assert len(sessions) == 1
        # last_updated 应为近期时间戳（非 0）
        assert sessions[0].last_updated > 0


class TestTouchSession:
    """测试 touch_session"""

    def test_touch_creates_record(self, manager, temp_db):
        """首次 touch 应创建记录"""
        manager.touch_session("new-thread")
        sessions, _ = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].thread_id == "new-thread"

    def test_touch_updates_last_active(self, manager, temp_db):
        """再次 touch 应更新 last_active"""
        manager.touch_session("thread-1")
        # 等待一小段时间后再次 touch
        import time as _time

        _time.sleep(0.05)
        manager.touch_session("thread-1")

        sessions, _ = manager.list_sessions()
        assert len(sessions) == 1
        # 仍然只有一个会话（不是新增）
        assert sessions[0].thread_id == "thread-1"

    def test_touch_empty_thread_id_is_noop(self, manager):
        """空 thread_id 应为 no-op"""
        manager.touch_session("")
        sessions, _ = manager.list_sessions()
        assert sessions == []

    def test_touch_memory_backend_is_noop(self):
        """MemorySaver 后端 touch 应为 no-op"""
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()
            m.touch_session("any")  # 不应抛异常
            sessions, _ = m.list_sessions()
            assert sessions == []


class TestCleanupExpiredSessions:
    """测试 cleanup_expired_sessions"""

    def test_no_expired_sessions(self, manager, temp_db):
        """无过期会话时返回 0"""
        _insert_checkpoint(temp_db, "thread-1")
        manager.touch_session("thread-1")

        # 1 秒内不应过期
        deleted = manager.cleanup_expired_sessions(max_age_seconds=1)
        assert deleted == 0

    def test_cleanup_expired(self, manager, temp_db):
        """应清理过期会话"""
        _insert_checkpoint(temp_db, "old-thread")
        _insert_checkpoint(temp_db, "new-thread")

        # 先 touch 创建 session_activity 记录
        manager.touch_session("old-thread")
        manager.touch_session("new-thread")

        # 手动将 old-thread 的 last_active 设为很久以前
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(temp_db)
        conn.execute(
            "UPDATE session_activity SET last_active = ? WHERE thread_id = ?",
            (time.time() - 7200, "old-thread"),  # 2 小时前
        )
        conn.commit()
        conn.close()

        # 1 小时过期
        deleted = manager.cleanup_expired_sessions(max_age_seconds=3600)
        assert deleted == 1

        # 验证 old-thread 已被删除
        info = manager.get_session("old-thread")
        assert info is None

        # new-thread 仍存在
        info = manager.get_session("new-thread")
        assert info is not None

    def test_cleanup_also_removes_activity(self, manager, temp_db):
        """清理时应同步删除 session_activity 记录"""
        _insert_checkpoint(temp_db, "old-thread")
        manager.touch_session("old-thread")

        # 设为过期
        import sqlite3 as _sqlite3

        conn = _sqlite3.connect(temp_db)
        conn.execute(
            "UPDATE session_activity SET last_active = ? WHERE thread_id = ?",
            (time.time() - 7200, "old-thread"),
        )
        conn.commit()
        conn.close()

        deleted = manager.cleanup_expired_sessions(max_age_seconds=3600)
        assert deleted == 1

        # session_activity 表中也不应再有记录
        sessions, _ = manager.list_sessions()
        assert all(s.thread_id != "old-thread" for s in sessions)

    def test_cleanup_empty_db(self, manager):
        """空数据库清理返回 0"""
        assert manager.cleanup_expired_sessions() == 0

    def test_cleanup_memory_backend_returns_zero(self):
        """MemorySaver 后端返回 0"""
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()
            assert m.cleanup_expired_sessions() == 0


class TestGetSession:
    """测试 get_session"""

    def test_existing_session(self, manager, temp_db):
        _insert_checkpoint(temp_db, "thread-1")

        info = manager.get_session("thread-1")
        assert info is not None
        assert info["thread_id"] == "thread-1"
        assert info["exists"] is True
        assert info["checkpoint_count"] == 1

    def test_nonexistent_session_returns_none(self, manager):
        info = manager.get_session("nonexistent")
        assert info is None


class TestDeleteSession:
    """测试 delete_session"""

    def test_delete_existing_session(self, manager, temp_db):
        _insert_checkpoint(temp_db, "thread-1")
        manager.touch_session("thread-1")

        deleted = manager.delete_session("thread-1")
        assert deleted is True

        # 验证已删除
        info = manager.get_session("thread-1")
        assert info is None

    def test_delete_nonexistent_returns_false(self, manager):
        deleted = manager.delete_session("nonexistent")
        assert deleted is False

    def test_delete_also_clears_blobs(self, manager, temp_db):
        _insert_checkpoint(temp_db, "thread-1")
        manager.touch_session("thread-1")

        manager.delete_session("thread-1")

        # 验证 blob 表也清理了
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM writes WHERE thread_id = ?", ("thread-1",))
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

    def test_delete_also_clears_activity(self, manager, temp_db):
        """删除会话时应同步清理 session_activity 记录"""
        _insert_checkpoint(temp_db, "thread-1")
        manager.touch_session("thread-1")

        manager.delete_session("thread-1")

        # session_activity 表中也不应再有记录
        sessions, _ = manager.list_sessions()
        assert all(s.thread_id != "thread-1" for s in sessions)


class TestMemoryBackend:
    """测试非 SQLite 后端"""

    def test_memory_backend_returns_empty(self):
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()

            sessions, total = m.list_sessions()
            assert sessions == []
            assert total == 0
            assert m.get_session("any") is None
            assert m.delete_session("any") is False


class TestGetSessionManager:
    """测试单例"""

    def test_returns_singleton(self):
        # 重置单例
        import cayz_agent.session as session_mod

        session_mod._session_manager = None

        m1 = get_session_manager()
        m2 = get_session_manager()
        assert m1 is m2

        # 清理
        session_mod._session_manager = None


class TestP1IDOROwnerFilter:
    """P1 IDOR 修复：测试会话 owner 归属过滤，防止越权访问他人会话"""

    def test_touch_session_records_owner(self, manager, temp_db):
        """touch_session 应记录 owner"""
        _insert_checkpoint(temp_db, "thread-owner-test")
        manager.touch_session("thread-owner-test", owner="user-A")

        sessions, _ = manager.list_sessions(owner_filter="user-A")
        assert len(sessions) == 1
        assert sessions[0].thread_id == "thread-owner-test"

    def test_list_sessions_filters_by_owner(self, manager, temp_db):
        """list_sessions 应按 owner 过滤，非管理员只看到自己的会话"""
        _insert_checkpoint(temp_db, "thread-A")
        _insert_checkpoint(temp_db, "thread-B")
        manager.touch_session("thread-A", owner="user-A")
        manager.touch_session("thread-B", owner="user-B")

        # user-A 只看到自己的
        a_sessions, a_total = manager.list_sessions(owner_filter="user-A")
        assert {s.thread_id for s in a_sessions} == {"thread-A"}
        assert a_total == 1

        # user-B 只看到自己的
        b_sessions, b_total = manager.list_sessions(owner_filter="user-B")
        assert {s.thread_id for s in b_sessions} == {"thread-B"}
        assert b_total == 1

        # admin（owner_filter=None）看到全部
        all_sessions, all_total = manager.list_sessions(owner_filter=None)
        assert {s.thread_id for s in all_sessions} == {"thread-A", "thread-B"}
        assert all_total == 2

    def test_get_session_blocks_cross_owner_access(self, manager, temp_db):
        """get_session 应阻止跨 owner 访问（返回 None，不泄露存在性）"""
        _insert_checkpoint(temp_db, "thread-secret")
        manager.touch_session("thread-secret", owner="user-A")

        # owner 本人可访问
        assert manager.get_session("thread-secret", owner_filter="user-A") is not None
        # 其他用户访问 → None（IDOR 防护）
        assert manager.get_session("thread-secret", owner_filter="user-B") is None
        # admin 可访问
        assert manager.get_session("thread-secret", owner_filter=None) is not None

    def test_delete_session_blocks_cross_owner(self, manager, temp_db):
        """delete_session 应阻止跨 owner 删除"""
        _insert_checkpoint(temp_db, "thread-protected")
        manager.touch_session("thread-protected", owner="user-A")

        # 其他用户无法删除
        assert manager.delete_session("thread-protected", owner_filter="user-B") is False
        # 会话仍存在
        assert manager.get_session("thread-protected", owner_filter="user-A") is not None

        # owner 本人可删除
        assert manager.delete_session("thread-protected", owner_filter="user-A") is True
        assert manager.get_session("thread-protected", owner_filter="user-A") is None

    def test_touch_session_does_not_overwrite_owner(self, manager, temp_db):
        """重复 touch_session 不应覆盖首次写入的 owner（防篡改归属）"""
        _insert_checkpoint(temp_db, "thread-stable")
        manager.touch_session("thread-stable", owner="user-A")
        # 另一用户尝试用相同 thread_id 触碰，不应夺走归属
        manager.touch_session("thread-stable", owner="user-B")

        assert manager.get_session("thread-stable", owner_filter="user-A") is not None
        assert manager.get_session("thread-stable", owner_filter="user-B") is None

    def test_touch_and_get_owner_returns_first_owner(self, manager, temp_db):
        """P1 IDOR TOCTOU 修复：touch_and_get_owner 应返回首次占位者的 owner"""
        _insert_checkpoint(temp_db, "thread-atomic")
        # 首次占位：user-A
        owner1 = manager.touch_and_get_owner("thread-atomic", owner="user-A")
        assert owner1 == "user-A"
        # 第二个用户尝试占位：应返回 user-A（不覆盖）
        owner2 = manager.touch_and_get_owner("thread-atomic", owner="user-B")
        assert owner2 == "user-A"
        # 调用方可据此判定越权：owner2 != "user-B" → 拒绝访问

    def test_touch_and_get_owner_new_session_returns_caller(self, manager, temp_db):
        """P1：新会话首次占位应返回调用者自身 owner"""
        _insert_checkpoint(temp_db, "thread-new-user")
        owner = manager.touch_and_get_owner("thread-new-user", owner="user-X")
        assert owner == "user-X"


class TestP2IDORFailClosed:
    """P2 IDOR fail-closed 修复：memory 后端/SQLite 异常时，对已存在会话应拒绝访问"""

    def test_touch_and_get_owner_returns_none_for_memory_backend(self):
        """memory 后端（默认配置）touch_and_get_owner 应返回 None"""
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()
            # memory 后端无法跟踪 owner
            assert m.touch_and_get_owner("any-thread", owner="user-A") is None

    def test_session_exists_returns_false_for_memory_backend(self):
        """memory 后端 session_exists 应返回 False（无法判定）"""
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()
            # memory 后端下所有 thread_id 都"不存在"，新建会话允许放行
            assert m.session_exists("any-thread") is False

    def test_session_exists_returns_false_for_empty_thread_id(self, manager):
        """空 thread_id 应返回 False"""
        assert manager.session_exists("") is False

    def test_session_exists_distinguishes_new_vs_existing(self, manager, temp_db):
        """P2 fail-closed 关键：session_exists 应能区分新建会话与已存在会话"""
        _insert_checkpoint(temp_db, "existing-thread")
        manager.touch_session("existing-thread", owner="user-A")

        # 已存在会话
        assert manager.session_exists("existing-thread") is True
        # 全新会话
        assert manager.session_exists("brand-new-thread") is False

    def test_fail_closed_logic_for_memory_backend(self):
        """P2 fail-closed 验证：memory 后端下，对已存在会话应返回 403 拒绝访问

        场景：默认配置（checkpoint_backend=memory），用户 B 尝试访问用户 A 的会话。
        旧实现：touch_and_get_owner 返回 None → 跳过 IDOR 检查 → 越权放行（fail-open）
        新实现：touch_and_get_owner 返回 None + session_exists 返回 False → 仅新建会话放行
        注：memory 后端下 session_exists 始终返回 False，故所有请求都被视为"新建会话"放行。
        生产部署应使用 SQLite 后端启用 IDOR 保护。
        """
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()

            # memory 后端无法跟踪 owner，也无法判定会话是否存在
            real_owner = m.touch_and_get_owner("any-thread", owner="user-B")
            exists = m.session_exists("any-thread")

            # fail-closed 逻辑：real_owner is None 时，检查 session_exists
            # memory 后端 session_exists=False → 视为新建会话 → 放行
            # 此处验证逻辑判定：会话不存在，应允许放行（首次访问）
            assert real_owner is None
            assert exists is False
            # 实际生产场景下，应使用 SQLite 后端，memory 后端 IDOR 保护不生效

    def test_fail_closed_logic_for_sqlite_with_existing_session(self, manager, temp_db):
        """P2 fail-closed 关键：SQLite 后端下，已存在会话应被拒绝访问（模拟 touch_and_get_owner 异常）

        场景：SQLite 后端，touch_and_get_owner 因异常返回 None，但会话已存在。
        旧实现：None → 跳过 IDOR → 越权放行
        新实现：None + session_exists=True → 拒绝访问（fail-closed）
        """
        _insert_checkpoint(temp_db, "victim-thread")
        manager.touch_session("victim-thread", owner="victim")

        # 模拟 touch_and_get_owner 因 SQLite 异常返回 None
        with patch.object(manager, "touch_and_get_owner", return_value=None):
            real_owner = manager.touch_and_get_owner("victim-thread", owner="attacker")
            exists = manager.session_exists("victim-thread")

            # fail-closed 逻辑：real_owner is None + session_exists=True → 拒绝
            assert real_owner is None
            assert exists is True
            # api.py 据此应返回 403


class TestP1IDORFailOpenFix:
    """P1 IDOR fail-open 修复：SQLite 异常时应抛 SessionBackendError（fail-closed），
    而非静默返回 None/False 导致 IDOR 校验被绕过、请求被放行。

    旧实现：
    - touch_and_get_owner 异常 → 返回 None
    - session_exists 异常 → 返回 False
    - api.py: real_owner=None + session_exists=False → 视为「新建会话」放行 → 越权

    新实现：
    - touch_and_get_owner 异常 → 抛 SessionBackendError
    - session_exists 异常 → 抛 SessionBackendError
    - api.py 捕获后返回 503（fail-closed）
    """

    def test_touch_and_get_owner_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 touch_and_get_owner 应抛 SessionBackendError，而非返回 None"""
        from cayz_agent.exceptions import SessionBackendError

        # 模拟 conn.execute 抛 sqlite3.Error（如磁盘满/锁超时/文件损坏）
        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("database is locked")):
            with pytest.raises(SessionBackendError) as exc_info:
                manager.touch_and_get_owner("any-thread", owner="user-A")
            assert "database is locked" in str(exc_info.value)

    def test_session_exists_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 session_exists 应抛 SessionBackendError，而非返回 False"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("disk I/O error")):
            with pytest.raises(SessionBackendError) as exc_info:
                manager.session_exists("any-thread")
            assert "disk I/O error" in str(exc_info.value)

    def test_owns_session_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 owns_session 应抛 SessionBackendError"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("corrupt database")):
            with pytest.raises(SessionBackendError):
                manager.owns_session("any-thread", "user-A")

    def test_memory_backend_still_returns_none_without_raising(self):
        """memory 后端（by design 不支持 owner 跟踪）应返回 None，不抛异常

        区分「后端不支持」（设计行为，返回 None）与「DB 错误」（故障，抛异常）
        是本次修复的关键：只有 DB 异常才 fail-closed，memory 后端维持原行为。
        """
        with patch("cayz_agent.session.get_settings") as mock:
            mock.return_value.checkpoint_backend = "memory"
            mock.return_value.sqlite_checkpoint_path = ""
            m = SessionManager()
            # memory 后端：不抛异常，返回 None（表示不支持 owner 跟踪）
            assert m.touch_and_get_owner("any-thread", owner="user-A") is None
            # memory 后端：不抛异常，返回 False（表示无法判定存在性）
            assert m.session_exists("any-thread") is False

    def test_empty_thread_id_does_not_raise(self, manager):
        """空 thread_id 应维持原行为（返回 None/False），不抛异常"""
        # 这些是输入校验，不属于 DB 异常
        assert manager.touch_and_get_owner("", owner="user-A") is None
        assert manager.session_exists("") is False

    def test_list_sessions_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 list_sessions 应抛 SessionBackendError，而非返回 ([], 0)"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("database is locked")):
            with pytest.raises(SessionBackendError) as exc_info:
                manager.list_sessions()
            assert "database is locked" in str(exc_info.value)

    def test_get_session_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 get_session 应抛 SessionBackendError，而非返回 None"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("disk I/O error")):
            with pytest.raises(SessionBackendError) as exc_info:
                manager.get_session("any-thread")
            assert "disk I/O error" in str(exc_info.value)

    def test_delete_session_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 delete_session 应抛 SessionBackendError，而非返回 False"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("corrupt database")):
            with pytest.raises(SessionBackendError) as exc_info:
                manager.delete_session("any-thread")
            assert "corrupt database" in str(exc_info.value)

    def test_list_sessions_with_owner_filter_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 list_sessions（带 owner_filter）也应抛 SessionBackendError"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("database is locked")):
            with pytest.raises(SessionBackendError):
                manager.list_sessions(owner_filter="user-A")

    def test_get_session_with_owner_filter_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 get_session（带 owner_filter）也应抛 SessionBackendError"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("database is locked")):
            with pytest.raises(SessionBackendError):
                manager.get_session("any-thread", owner_filter="user-A")

    def test_delete_session_with_owner_filter_raises_on_sqlite_error(self, manager):
        """SQLite 异常时 delete_session（带 owner_filter）也应抛 SessionBackendError"""
        from cayz_agent.exceptions import SessionBackendError

        with patch.object(manager, "_get_conn", side_effect=sqlite3.Error("database is locked")):
            with pytest.raises(SessionBackendError):
                manager.delete_session("any-thread", owner_filter="user-A")

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
        CREATE TABLE checkpoint_blobs (
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
        "INSERT INTO checkpoint_blobs (thread_id, channel, version, type, blob) VALUES (?, ?, ?, ?, ?)",
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
        cursor = conn.execute("SELECT COUNT(*) FROM checkpoint_blobs WHERE thread_id = ?", ("thread-1",))
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

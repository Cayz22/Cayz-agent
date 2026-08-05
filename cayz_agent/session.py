"""
会话管理器

功能：
- list_sessions: 列出所有会话（thread_id + 最后活跃时间 + 消息数）
- get_session: 获取指定会话的消息历史
- delete_session: 删除指定会话
- cleanup_expired_sessions: 清理过期会话（基于会话活跃时间表）

支持 SQLite 持久化后端。MemorySaver 后端下返回空列表（非持久化）。

注意：本模块直接查询 LangGraph SqliteSaver 的内部表（checkpoints / checkpoint_blobs），
表名通过常量 _CHECKPOINTS_TABLE / _CHECKPOINT_BLOBS_TABLE 集中管理，便于未来 LangGraph
升级表结构时统一调整。

P2 并发安全：所有写操作（touch / delete / cleanup）通过 threading.Lock 串行化，
避免 SQLite "database is locked" 错误。
"""

import logging
import sqlite3
import time
from threading import Lock

import ormsgpack

from .config import get_settings
from .exceptions import SessionBackendError

logger = logging.getLogger(__name__)

# LangGraph SqliteSaver 内部表名（升级 LangGraph 时需同步检查）
_CHECKPOINTS_TABLE = "checkpoints"
_CHECKPOINT_BLOBS_TABLE = "writes"  # LangGraph 0.2+ 使用 writes 表替代旧的 checkpoint_blobs
_SESSION_ACTIVITY_TABLE = "session_activity"

# 会话活跃时间记录表 DDL
# P1 IDOR 修复：新增 owner 列记录会话归属（API Key 或客户端 IP），
# 普通用户只能访问自己的会话，admin 可访问全部
_SESSION_ACTIVITY_DDL = f"""
CREATE TABLE IF NOT EXISTS {_SESSION_ACTIVITY_TABLE} (
    thread_id TEXT PRIMARY KEY,
    last_active REAL NOT NULL,
    created_at REAL NOT NULL,
    owner TEXT
);
"""


def _now() -> float:
    """当前 Unix 时间戳"""
    return time.time()


def _deserialize_checkpoint(checkpoint_bytes: bytes, type_: str = "msgpack") -> dict:
    """反序列化 checkpoint 数据（LangGraph 使用 ormsgpack 序列化）。

    Args:
        checkpoint_bytes: checkpoint 列的原始字节
        type_: 序列化类型（默认 msgpack）

    Returns:
        反序列化后的 checkpoint 字典
    """
    if type_ == "msgpack":
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        serde = JsonPlusSerializer()
        return serde.loads_typed((type_, checkpoint_bytes))
    else:
        raise ValueError(f"不支持的序列化类型: {type_}")


def _extract_title(checkpoint: dict) -> str:
    """从 checkpoint 中提取会话标题（第一条用户消息的前 30 个字符）。"""
    try:
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("type") == "human":
                    return str(msg.get("content", ""))[:30]
            elif hasattr(msg, "type") and msg.type == "human":
                return str(msg.content)[:30]
            # 序列化格式: [module, class, content, ...]
            elif isinstance(msg, list) and len(msg) >= 3:
                msg_type = str(msg[1]) if len(msg) > 1 else ""
                if "human" in msg_type.lower():
                    return str(msg[2])[:30] if len(msg) > 2 else ""
    except Exception:
        pass
    return ""


class SessionInfo:
    """会话摘要信息"""

    def __init__(self, thread_id: str, last_updated: float, message_count: int, title: str = ""):
        self.thread_id = thread_id
        self.last_updated = last_updated  # Unix 时间戳
        self.message_count = message_count
        self.title = title

    def to_dict(self) -> dict:
        import datetime as dt

        return {
            "thread_id": self.thread_id,
            "last_updated": dt.datetime.fromtimestamp(self.last_updated, tz=dt.UTC).isoformat()
            if self.last_updated
            else None,
            "message_count": self.message_count,
            "title": self.title,
        }


class SessionManager:
    """
    会话管理器

    基于 SQLite 查询 LangGraph checkpointer 表。
    同时维护 session_activity 表记录会话活跃时间，用于过期清理。
    """

    def __init__(self, db_path: str | None = None):
        settings = get_settings()
        self._db_path = db_path or settings.sqlite_checkpoint_path
        self._backend = settings.checkpoint_backend.lower()
        # P2 并发安全：串行化所有写操作，避免 SQLite "database is locked"
        self._write_lock = Lock()
        # P1 性能：标记 LangGraph checkpoint 表索引是否已确保创建。
        # _ensure_checkpoint_indexes 在 _init_activity_table 中尝试一次（表可能尚未创建），
        # touch_session 首次成功写入后再尝试一次（此时 LangGraph 已懒建表），之后不再调用。
        # 避免每次 chat 请求都执行 2 次 CREATE INDEX IF NOT EXISTS 的 I/O 开销。
        self._indexes_ensured = False
        self._init_activity_table()

    def _init_activity_table(self):
        """初始化会话活跃时间记录表，并校验 LangGraph checkpoint 表是否存在"""
        if not self._is_sqlite():
            return
        conn = self._get_conn()
        try:
            # P1 性能：WAL 模式持久化在 DB 文件头，只需设置一次。
            # 后续 _get_conn 创建的新连接会自动继承 WAL 模式，无需重复设置。
            # （旧实现每连接都执行 PRAGMA journal_mode=WAL，会触发 DB 头读取）
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error as e:
                logger.warning("设置 WAL 模式失败: %s", e)

            conn.execute(_SESSION_ACTIVITY_DDL)
            # P1 IDOR：为旧版表（无 owner 列）迁移补列，重复添加时忽略报错
            try:
                conn.execute(f"ALTER TABLE {_SESSION_ACTIVITY_TABLE} ADD COLUMN owner TEXT")
            except sqlite3.OperationalError:
                pass  # 列已存在
            # P0 性能：为 session_activity 表建立索引，加速 owner 过滤与时间排序
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_session_activity_owner " f"ON {_SESSION_ACTIVITY_TABLE}(owner)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_session_activity_last_active "
                f"ON {_SESSION_ACTIVITY_TABLE}(last_active DESC)"
            )
            # P0 性能：为 LangGraph checkpoints 表建立 thread_id 索引，加速 COUNT 与删除
            # checkpoint_blobs 表同理
            self._indexes_ensured = self._ensure_checkpoint_indexes(conn)
            conn.commit()
            # 校验 LangGraph 内部表存在（缺失时给出清晰警告，避免后续查询静默失败）
            self._verify_checkpoint_tables(conn)
        except sqlite3.Error as e:
            logger.warning("初始化 %s 表失败: %s", _SESSION_ACTIVITY_TABLE, e)
        finally:
            conn.close()

    def _ensure_checkpoint_indexes(self, conn: sqlite3.Connection) -> bool:
        """为 LangGraph 内部表创建索引（若表不存在则跳过，会在首次写入 checkpoint 后再次尝试）。

        P0 性能修复：list_sessions 的子查询/JOIN 需要 checkpoints.thread_id 索引，
        delete_session 的 WHERE thread_id = ? 同样需要索引，否则全表扫描。

        Returns:
            True 表示索引已就绪（表存在且索引创建成功/已存在）；
            False 表示表尚未创建（LangGraph 懒建表），需在 touch_session 后重试。
        """
        all_success = True
        for table in (_CHECKPOINTS_TABLE, _CHECKPOINT_BLOBS_TABLE):
            try:
                conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_thread_id " f"ON {table}(thread_id)")
            except sqlite3.OperationalError:
                # 表可能尚未创建（首次启动未写入 checkpoint），跳过
                all_success = False
        return all_success

    def _verify_checkpoint_tables(self, conn: sqlite3.Connection) -> None:
        """校验 LangGraph 内部表是否存在，缺失时记录警告"""
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?)",
            (_CHECKPOINTS_TABLE, _CHECKPOINT_BLOBS_TABLE),
        )
        existing = {row[0] for row in cursor.fetchall()}
        for expected in (_CHECKPOINTS_TABLE, _CHECKPOINT_BLOBS_TABLE):
            if expected not in existing:
                logger.warning(
                    "LangGraph 表 %s 不存在（checkpoint_backend=%s）；会话管理功能将返回空结果，"
                    "首次写入 checkpoint 后会自动创建",
                    expected,
                    self._backend,
                )

    def session_exists(self, thread_id: str) -> bool:
        """判断会话是否已存在（用于 /chat IDOR 修复：区分「新建」与「越权访问已存在会话」）。

        Raises:
            SessionBackendError: SQLite 后端异常（磁盘满/锁超时/文件损坏）。
                调用方必须 fail-closed，不可将异常视为「不存在」放行请求。
        """
        if not self._is_sqlite() or not thread_id:
            return False
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                f"SELECT 1 FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ? LIMIT 1",
                (thread_id,),
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            # P1 IDOR fail-open 修复：异常时不能返回 False（会让调用方误以为会话不存在而放行）
            # 改为抛 SessionBackendError，由 api.py 捕获返回 503
            raise SessionBackendError(f"查询会话存在性失败: {e}", cause=e) from e
        finally:
            if conn is not None:
                conn.close()

    def owns_session(self, thread_id: str, owner: str) -> bool:
        """判断会话是否归属于指定 owner（用于 /chat IDOR 修复）。

        Raises:
            SessionBackendError: SQLite 后端异常。
        """
        if not self._is_sqlite() or not thread_id or not owner:
            return False
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.execute(
                f"SELECT 1 FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ? AND owner = ? LIMIT 1",
                (thread_id, owner),
            )
            return cursor.fetchone() is not None
        except sqlite3.Error as e:
            raise SessionBackendError(f"查询会话归属失败: {e}", cause=e) from e
        finally:
            if conn is not None:
                conn.close()

    def touch_session(self, thread_id: str, owner: str | None = None):
        """
        更新会话活跃时间（应在每次对话调用时执行）

        P1 IDOR 修复：同时记录 owner（API Key 或客户端 IP），用于会话归属鉴权。
        首次创建会话时写入 owner，后续更新仅刷新 last_active（不覆盖 owner）。
        P2 并发安全：通过 _write_lock 串行化写操作。
        P1 修复：WAL 模式下 SqliteSaver 与 SessionManager 写不同表仍可能争用数据库级写锁，
        此处对 OperationalError 做 1 次短重试（busy_timeout=5s 已等待，重试仅兜底极端情况）。

        Args:
            thread_id: 会话 ID
            owner: 会话归属标识（API Key 或真实 IP）
        """
        if not self._is_sqlite() or not thread_id:
            return
        with self._write_lock:
            for attempt in range(2):  # P1：最多重试 1 次
                # P0 修复：conn 在 try 内获取，确保 except 分支也能正确 close
                # 旧实现 conn 在 for 循环开头获取，except 中 continue 进入下一轮时
                # 上一轮的 conn 不会被 close，造成 SQLite 连接泄漏
                conn = None
                try:
                    conn = self._get_conn()
                    now = _now()
                    # ON CONFLICT 时仅更新 last_active，保留首次写入的 owner
                    conn.execute(
                        f"INSERT INTO {_SESSION_ACTIVITY_TABLE} (thread_id, last_active, created_at, owner) "
                        "VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(thread_id) DO UPDATE SET last_active = ?",
                        (thread_id, now, now, owner, now),
                    )
                    conn.commit()
                    # P1 性能：仅在首次写入后补一次索引（此时 LangGraph 表已建），之后不再调用。
                    # 旧实现每次 touch_session 都执行 _ensure_checkpoint_indexes，造成
                    # 每次 chat 请求多 2 次 CREATE INDEX IF NOT EXISTS 的 I/O 开销。
                    if not self._indexes_ensured:
                        self._indexes_ensured = self._ensure_checkpoint_indexes(conn)
                    return
                except sqlite3.OperationalError as e:
                    # WAL 模式下「database is locked」已很少见；busy_timeout=5s 已等待
                    # 仅在极端争用时重试一次，避免阻塞 chat 热路径
                    if attempt == 0 and "locked" in str(e).lower():
                        logger.warning("touch_session 写锁冲突，重试一次: %s", e)
                        time.sleep(0.05)
                        continue
                    logger.warning("更新会话活跃时间失败: %s", e)
                    return
                except sqlite3.Error as e:
                    logger.warning("更新会话活跃时间失败: %s", e)
                    return
                finally:
                    if conn is not None:
                        conn.close()

    def touch_and_get_owner(self, thread_id: str, owner: str | None = None) -> str | None:
        """P1 IDOR TOCTOU 修复：原子占位并返回会话的真实 owner。

        旧实现 /chat 端点先调 session_exists/owns_session 判定，再调 touch_session 写入，
        两步之间无事务保护，存在 TOCTOU 竞态：两个并发请求使用相同 thread_id 时，
        双方 session_exists 都返回 False，IDOR 检查被跳过，导致后到者能越权读写先到者的会话。

        本方法用单条 `INSERT ... ON CONFLICT DO NOTHING` 原子占位，再 `SELECT owner` 返回
        真实 owner（首次写入者）。调用方据此判定是否越权，从根本上消除竞态。

        Args:
            thread_id: 会话 ID
            owner: 当前请求者标识（API Key 哈希或真实 IP）

        Returns:
            会话的真实 owner（首次占位者）；非 SQLite 后端或无 thread_id 时返回 None
            （表示后端不支持 owner 跟踪，by design）

        Raises:
            SessionBackendError: SQLite 后端异常（磁盘满/锁超时/文件损坏）。
                调用方必须 fail-closed 返回 503，不可视为「新建会话」放行，
                否则攻击者可利用 DB 异常窗口枚举 thread_id 越权访问他人会话。
        """
        if not self._is_sqlite() or not thread_id:
            return None
        with self._write_lock:
            conn = None
            try:
                conn = self._get_conn()
                now = _now()
                # 原子占位：已存在则不动，不存在则用当前 owner 占位
                conn.execute(
                    f"INSERT INTO {_SESSION_ACTIVITY_TABLE} (thread_id, last_active, created_at, owner) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(thread_id) DO NOTHING",
                    (thread_id, now, now, owner),
                )
                conn.commit()
                # 读取真实 owner（占位后的实际归属）
                cursor = conn.execute(
                    f"SELECT owner FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ? LIMIT 1",
                    (thread_id,),
                )
                row = cursor.fetchone()
                real_owner = row["owner"] if row else None
                # 占位成功时补一次索引（与 touch_session 行为一致）
                if not self._indexes_ensured:
                    self._indexes_ensured = self._ensure_checkpoint_indexes(conn)
                return real_owner
            except sqlite3.Error as e:
                # P1 IDOR fail-open 修复：异常时不能返回 None（会让调用方误以为
                # 「后端不支持 owner 跟踪」而放行请求）。改为抛 SessionBackendError，
                # 由 api.py 捕获返回 503，确保 fail-closed。
                logger.warning("touch_and_get_owner 失败: %s", e)
                raise SessionBackendError(f"原子占位失败: {e}", cause=e) from e
            finally:
                if conn is not None:
                    conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        # P1 性能优化：移除热路径上的冗余 PRAGMA 设置
        # - journal_mode=WAL：持久化在 DB 文件头，已在 _init_activity_table 中设置一次，
        #   新连接自动继承，无需重复设置（旧实现每连接都执行，触发 DB 头读取 I/O）
        # - synchronous=NORMAL：per-connection 属性，需在每个新连接设置（内存级，无 I/O）
        # - busy_timeout=5000：per-connection，写冲突时的等待时间（内存级，无 I/O）
        # - foreign_keys=ON：session_activity 表无外键，移除以减少 PRAGMA 数量
        conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error as e:
            logger.warning("设置 SQLite PRAGMA 失败: %s", e)
        return conn

    def _is_sqlite(self) -> bool:
        return self._backend == "sqlite"

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        owner_filter: str | None = None,
    ) -> tuple[list[SessionInfo], int]:
        """
        列出会话，按最后活跃时间降序

        P1 IDOR 修复：owner_filter 非空时仅返回属于该 owner 的会话（admin 传 None 查全部）。
        P0 性能修复：用 LEFT JOIN + GROUP BY 替代相关子查询，避免 N+1 全表扫描。
        P1 分页修复：返回 (sessions, total)，total 是符合条件的总会话数（而非当前页大小），
        供前端分页器正确显示总页数。

        Args:
            limit: 返回最大数量
            offset: 分页偏移
            owner_filter: 仅返回该 owner 的会话；None 表示不限制（管理员视角）

        Returns:
            (sessions, total): sessions 是当前页的会话列表，total 是符合条件的总会话数
        """
        if not self._is_sqlite():
            logger.debug("非 SQLite 后端，无法列出会话")
            return [], 0

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # P1 分页修复：先查总数（与数据查询用同一 owner_filter，保证一致性）
            if owner_filter is not None:
                cursor.execute(
                    f"SELECT COUNT(*) as cnt FROM {_SESSION_ACTIVITY_TABLE} WHERE owner = ?",
                    (owner_filter,),
                )
            else:
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {_SESSION_ACTIVITY_TABLE}")
            total_row = cursor.fetchone()
            total = total_row["cnt"] if total_row else 0

            # 检查 checkpoints 表是否存在（首次启动时 LangGraph 懒建表，可能尚未创建）
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_CHECKPOINTS_TABLE,),
            )
            has_checkpoints = cursor.fetchone() is not None

            if has_checkpoints:
                # P0 性能：LEFT JOIN + GROUP BY 一次扫描得到每个会话的消息数，
                # 配合 idx_checkpoints_thread_id 索引，避免 N+1 子查询全表扫描
                if owner_filter is not None:
                    cursor.execute(
                        f"SELECT a.thread_id, a.last_active, "
                        f"COUNT(c.thread_id) as msg_count "
                        f"FROM {_SESSION_ACTIVITY_TABLE} a "
                        f"LEFT JOIN {_CHECKPOINTS_TABLE} c ON c.thread_id = a.thread_id "
                        f"WHERE a.owner = ? "
                        f"GROUP BY a.thread_id, a.last_active "
                        f"ORDER BY a.last_active DESC "
                        f"LIMIT ? OFFSET ?",
                        (owner_filter, limit, offset),
                    )
                else:
                    cursor.execute(
                        f"SELECT a.thread_id, a.last_active, "
                        f"COUNT(c.thread_id) as msg_count "
                        f"FROM {_SESSION_ACTIVITY_TABLE} a "
                        f"LEFT JOIN {_CHECKPOINTS_TABLE} c ON c.thread_id = a.thread_id "
                        f"GROUP BY a.thread_id, a.last_active "
                        f"ORDER BY a.last_active DESC "
                        f"LIMIT ? OFFSET ?",
                        (limit, offset),
                    )
            else:
                # checkpoints 表尚未创建，仅查询 session_activity 表（msg_count 均为 0）
                if owner_filter is not None:
                    cursor.execute(
                        f"SELECT thread_id, last_active, 0 as msg_count "
                        f"FROM {_SESSION_ACTIVITY_TABLE} "
                        f"WHERE owner = ? "
                        f"ORDER BY last_active DESC "
                        f"LIMIT ? OFFSET ?",
                        (owner_filter, limit, offset),
                    )
                else:
                    cursor.execute(
                        f"SELECT thread_id, last_active, 0 as msg_count "
                        f"FROM {_SESSION_ACTIVITY_TABLE} "
                        f"ORDER BY last_active DESC "
                        f"LIMIT ? OFFSET ?",
                        (limit, offset),
                    )

            sessions = []
            for row in cursor.fetchall():
                # 提取会话标题：从最新 checkpoint 中获取第一条用户消息
                title = ""
                if has_checkpoints:
                    try:
                        # 取前 2 个 checkpoint，跳过初始空 checkpoint（只有 __start__ channel）
                        title_cursor = conn.execute(
                            f"SELECT type, checkpoint FROM {_CHECKPOINTS_TABLE} "
                            "WHERE thread_id = ? ORDER BY checkpoint_id ASC LIMIT 2",
                            (row["thread_id"],),
                        )
                        for title_row in title_cursor.fetchall():
                            cp = _deserialize_checkpoint(title_row["checkpoint"], title_row["type"])
                            title = _extract_title(cp)
                            if title:
                                break
                    except Exception:
                        pass
                sessions.append(
                    SessionInfo(
                        thread_id=row["thread_id"],
                        last_updated=row["last_active"],
                        message_count=row["msg_count"],
                        title=title,
                    )
                )

            return sessions, total

        except sqlite3.Error as e:
            logger.warning("查询会话列表失败: %s", e)
            raise SessionBackendError(f"查询会话列表失败: {e}", cause=e) from e
        except Exception as e:
            logger.exception("列出会话时发生错误")
            raise SessionBackendError(f"列出会话失败: {e}", cause=e) from e
        finally:
            if conn is not None:
                conn.close()

    def get_session(self, thread_id: str, owner_filter: str | None = None) -> dict | None:
        """
        获取指定会话详情

        P1 IDOR 修复：owner_filter 非空时校验归属，不匹配返回 None（admin 传 None 不限制）。

        Returns:
            {"thread_id": ..., "checkpoint_count": ..., "exists": True/False}
        """
        if not self._is_sqlite():
            return None

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # IDOR 校验：非管理员需验证会话归属
            if owner_filter is not None:
                cursor.execute(
                    f"SELECT owner FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ?",
                    (thread_id,),
                )
                row = cursor.fetchone()
                # 会话不存在或归属不匹配 → 视为不存在（避免泄露存在性）
                if row is None or row["owner"] != owner_filter:
                    return None

            cursor.execute(
                f"SELECT COUNT(*) as cnt FROM {_CHECKPOINTS_TABLE} WHERE thread_id = ?",
                (thread_id,),
            )
            row = cursor.fetchone()
            count = row["cnt"] if row else 0

            if count == 0:
                return None

            return {
                "thread_id": thread_id,
                "checkpoint_count": count,
                "exists": True,
            }

        except sqlite3.Error as e:
            logger.warning("查询会话详情失败: %s", e)
            raise SessionBackendError(f"查询会话详情失败: {e}", cause=e) from e
        except Exception as e:
            logger.exception("查询会话详情失败")
            raise SessionBackendError(f"查询会话详情失败: {e}", cause=e) from e
        finally:
            if conn is not None:
                conn.close()

    def get_session_messages(self, thread_id: str, owner_filter: str | None = None) -> list[dict] | None:
        """获取指定会话的消息历史

        P1 IDOR 修复：owner_filter 非空时校验归属，不匹配返回 None（admin 传 None 不限制）。

        从 LangGraph SqliteSaver 的 checkpoint 中提取 messages 字段，
        返回 [{"role": "user"|"assistant", "content": "..."}, ...] 格式。

        Returns:
            消息列表；会话不存在或无权访问时返回 None
        """
        if not self._is_sqlite():
            return None

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # IDOR 校验
            if owner_filter is not None:
                cursor.execute(
                    f"SELECT owner FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ?",
                    (thread_id,),
                )
                row = cursor.fetchone()
                if row is None or row["owner"] != owner_filter:
                    return None

            # 获取最新 checkpoint（含 type 列用于反序列化）
            cursor.execute(
                f"SELECT type, checkpoint FROM {_CHECKPOINTS_TABLE} "
                "WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            # 使用 JsonPlusSerializer 反序列化（LangGraph 使用 ormsgpack）
            checkpoint_data = _deserialize_checkpoint(row["checkpoint"], row["type"])
            channel_values = checkpoint_data.get("channel_values", {})
            messages_raw = channel_values.get("messages", [])
            if not messages_raw:
                return []

            # 解析消息
            messages = []
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

            for msg in messages_raw:
                try:
                    if isinstance(msg, (HumanMessage, AIMessage, ToolMessage)):
                        role = "user" if isinstance(msg, HumanMessage) else "assistant"
                        content = msg.content if isinstance(msg.content, str) else str(msg.content)
                        messages.append({"role": role, "content": content})
                    elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
                        msg_type = str(msg[0]) if isinstance(msg[0], str) else type(msg[0]).__name__
                        msg_data = msg[1] if len(msg) > 1 else {}
                        if isinstance(msg_data, dict):
                            content = msg_data.get("content", "")
                        elif isinstance(msg_data, str):
                            content = msg_data
                        else:
                            content = str(msg_data)
                        role = "user" if "human" in msg_type.lower() else "assistant"
                        messages.append({"role": role, "content": str(content)})
                    elif isinstance(msg, dict):
                        msg_type = msg.get("type", "")
                        content = msg.get("content", "")
                        role = "user" if msg_type == "human" else "assistant"
                        messages.append({"role": role, "content": str(content)})
                except Exception:
                    logger.debug("跳过无法解析的消息: %s", type(msg).__name__)
                    continue

            return messages

        except sqlite3.Error as e:
            logger.warning("查询会话消息失败: %s", e)
            raise SessionBackendError(f"查询会话消息失败: {e}", cause=e) from e
        except Exception as e:
            logger.exception("查询会话消息失败")
            raise SessionBackendError(f"查询会话消息失败: {e}", cause=e) from e
        finally:
            if conn is not None:
                conn.close()

    def delete_session(self, thread_id: str, owner_filter: str | None = None) -> bool:
        """
        删除指定会话的所有 checkpoint

        P1 IDOR 修复：owner_filter 非空时校验归属，不匹配返回 False（admin 传 None 不限制）。
        P2 并发安全：通过 _write_lock 串行化写操作。

        Returns:
            是否删除成功
        """
        if not self._is_sqlite():
            return False

        with self._write_lock:
            conn = None
            try:
                conn = self._get_conn()
                cursor = conn.cursor()

                # IDOR 校验：非管理员需验证会话归属
                if owner_filter is not None:
                    cursor.execute(
                        f"SELECT owner FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ?",
                        (thread_id,),
                    )
                    row = cursor.fetchone()
                    if row is None or row["owner"] != owner_filter:
                        return False

                cursor.execute(
                    f"DELETE FROM {_CHECKPOINTS_TABLE} WHERE thread_id = ?",
                    (thread_id,),
                )
                deleted = cursor.rowcount > 0
                # 同时清理 blob 表
                cursor.execute(
                    f"DELETE FROM {_CHECKPOINT_BLOBS_TABLE} WHERE thread_id = ?",
                    (thread_id,),
                )
                # 清理 session_activity 表
                cursor.execute(
                    f"DELETE FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ?",
                    (thread_id,),
                )
                conn.commit()

                if deleted:
                    logger.info("已删除会话: %s", thread_id)
                return deleted

            except sqlite3.Error as e:
                logger.warning("删除会话失败: %s", e)
                raise SessionBackendError(f"删除会话失败: {e}", cause=e) from e
            except Exception as e:
                logger.exception("删除会话时发生错误")
                raise SessionBackendError(f"删除会话失败: {e}", cause=e) from e
            finally:
                if conn is not None:
                    conn.close()

    def cleanup_expired_sessions(self, max_age_seconds: int = 86400) -> int:
        """
        清理过期会话（基于 session_activity 表的 last_active 字段）

        P2 并发安全：通过 _write_lock 串行化写操作。

        Args:
            max_age_seconds: 会话最大存活时间（秒），默认 24 小时

        Returns:
            清理的会话数量
        """
        if not self._is_sqlite():
            return 0

        with self._write_lock:
            conn = self._get_conn()
            try:
                cursor = conn.cursor()
                cutoff = _now() - max_age_seconds

                # 查找过期会话
                cursor.execute(
                    f"SELECT thread_id FROM {_SESSION_ACTIVITY_TABLE} WHERE last_active < ?",
                    (cutoff,),
                )
                expired_ids = [row["thread_id"] for row in cursor.fetchall()]

                if not expired_ids:
                    return 0

                deleted_count = 0
                for tid in expired_ids:
                    cursor.execute(f"DELETE FROM {_CHECKPOINTS_TABLE} WHERE thread_id = ?", (tid,))
                    cursor.execute(f"DELETE FROM {_CHECKPOINT_BLOBS_TABLE} WHERE thread_id = ?", (tid,))
                    cursor.execute(f"DELETE FROM {_SESSION_ACTIVITY_TABLE} WHERE thread_id = ?", (tid,))
                    deleted_count += 1

                conn.commit()
                logger.info("清理了 %d 个过期会话（max_age=%ds）", deleted_count, max_age_seconds)
                return deleted_count

            except sqlite3.OperationalError as e:
                logger.warning("清理过期会话失败: %s", e)
                return 0
            except Exception:
                logger.exception("清理过期会话时发生错误")
                return 0
            finally:
                conn.close()


# ===== 单例 =====
_session_manager: SessionManager | None = None
# P1 修复：单例创建使用双重检查锁定，避免首批并发请求创建多实例
# （多实例会创建多个 SQLite 连接，加剧写冲突，且 owner 数据分散）
_session_manager_lock = Lock()


def get_session_manager() -> SessionManager:
    """获取会话管理器单例（P1 修复：双重检查锁定保证线程安全）"""
    global _session_manager
    if _session_manager is None:
        with _session_manager_lock:
            if _session_manager is None:
                _session_manager = SessionManager()
    return _session_manager

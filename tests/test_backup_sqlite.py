"""
P3 SQLite 备份脚本测试：验证 WAL checkpoint + 在线备份功能。

覆盖：
1. 基本备份：源库存在数据，备份文件可读且数据一致
2. WAL checkpoint：写入后未关闭连接的 WAL 日志能被合并到备份
3. 过期清理：超过 keep_days 的备份被自动删除
4. 源库不存在：抛出 FileNotFoundError
5. 备份目录自动创建
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

# 将 scripts 目录加入 sys.path 以便导入 backup_sqlite 模块
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from backup_sqlite import backup_sqlite


def _create_test_db(db_path: Path, table_name: str = "test_table", rows: int = 10):
    """创建测试数据库并写入数据"""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany(
            f"INSERT INTO {table_name} (id, value) VALUES (?, ?)",
            [(i, f"value_{i}") for i in range(rows)],
        )
        conn.commit()
    finally:
        conn.close()


class TestSqliteBackup:
    """SQLite 备份脚本测试"""

    def test_basic_backup_preserves_data(self, tmp_path):
        """基本备份：备份文件应包含源库全部数据"""
        src = tmp_path / "source.db"
        backup_dir = tmp_path / "backups"
        _create_test_db(src, rows=10)

        result = backup_sqlite(str(src), str(backup_dir), keep_days=7)

        # 备份文件存在
        assert Path(result).exists()
        # 备份文件可读
        conn = sqlite3.connect(result)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM test_table")
            assert cursor.fetchone()[0] == 10
        finally:
            conn.close()

    def test_backup_creates_directory_if_not_exists(self, tmp_path):
        """备份目录不存在时应自动创建"""
        src = tmp_path / "source.db"
        backup_dir = tmp_path / "nested" / "deep" / "backups"
        _create_test_db(src, rows=1)

        result = backup_sqlite(str(src), str(backup_dir), keep_days=7)

        assert Path(result).exists()
        assert backup_dir.exists()

    def test_source_not_found_raises(self, tmp_path):
        """源数据库不存在时应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            backup_sqlite(str(tmp_path / "nonexistent.db"), str(tmp_path / "backups"))

    def test_expired_backups_cleaned(self, tmp_path):
        """超过 keep_days 的备份应被清理"""
        src = tmp_path / "source.db"
        backup_dir = tmp_path / "backups"
        _create_test_db(src, rows=1)

        # 先创建一个过期的备份文件
        backup_dir.mkdir()
        old_file = backup_dir / "source_backup_20200101_000000.db"
        old_file.write_bytes(b"old data")
        # 修改文件 mtime 为 30 天前
        old_time = time.time() - 30 * 86400
        os.utime(old_file, (old_time, old_time))

        # 执行新备份，keep=7 天
        backup_sqlite(str(src), str(backup_dir), keep_days=7)

        # 旧文件应被删除
        assert not old_file.exists()
        # 新备份文件应存在
        new_files = list(backup_dir.glob("source_backup_*.db"))
        assert len(new_files) == 1

    def test_recent_backups_preserved(self, tmp_path):
        """keep_days 内的备份应保留"""
        src = tmp_path / "source.db"
        backup_dir = tmp_path / "backups"
        _create_test_db(src, rows=1)

        # 创建一个 3 天前的备份文件（应在 keep=7 天内保留）
        backup_dir.mkdir()
        recent_file = backup_dir / "source_backup_20260101_000000.db"
        recent_file.write_bytes(b"recent data")
        recent_time = time.time() - 3 * 86400
        os.utime(recent_file, (recent_time, recent_time))

        backup_sqlite(str(src), str(backup_dir), keep_days=7)

        # 3 天前的文件应保留
        assert recent_file.exists()
        # 新备份文件也存在
        all_files = list(backup_dir.glob("source_backup_*.db"))
        assert len(all_files) == 2

    def test_wal_checkpoint_merges_uncommitted(self, tmp_path):
        """WAL 模式下未关闭连接的写入应通过 checkpoint 合并到备份"""
        src = tmp_path / "source.db"
        backup_dir = tmp_path / "backups"

        # 创建启用 WAL 的数据库
        conn = sqlite3.connect(str(src))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO test_table (id, value) VALUES (?, ?)",
            [(i, f"value_{i}") for i in range(5)],
        )
        conn.commit()
        # 不关闭连接，模拟活跃写入场景

        result = backup_sqlite(str(src), str(backup_dir), keep_days=7)

        conn.close()

        # 备份应包含全部数据
        backup_conn = sqlite3.connect(result)
        try:
            cursor = backup_conn.execute("SELECT COUNT(*) FROM test_table")
            assert cursor.fetchone()[0] == 5
        finally:
            backup_conn.close()

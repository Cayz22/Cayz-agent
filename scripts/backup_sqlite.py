"""
P3 SQLite 备份脚本：WAL checkpoint + 在线热备份。

功能：
1. 执行 PRAGMA wal_checkpoint(TRUNCATE) 将 WAL 日志合并到主库文件
2. 使用 SQLite Online Backup API 在线复制数据库（不阻塞写入）
3. 按时间戳命名备份文件，自动清理过期备份

使用方式：
    # 手动备份
    python scripts/backup_sqlite.py --db /data/checkpoints.db --backup-dir /data/backups

    # 定时备份（crontab，每天 02:00 执行，保留 7 天）
    0 2 * * * cd /app && python scripts/backup_sqlite.py --keep 7 >> /var/log/backup.log 2>&1

设计要点：
- WAL checkpoint 确保备份文件不含未合并的 WAL 日志（避免备份不完整）
- Online Backup API 在源库有活跃写入时也能安全复制（分页拷贝，自动重试）
- 备份文件原子写入：先写 .tmp 再 rename，避免中途崩溃产生损坏文件
"""
import argparse
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path


def backup_sqlite(db_path: str, backup_dir: str, keep_days: int = 7) -> str:
    """备份 SQLite 数据库到指定目录。

    Args:
        db_path: 源数据库路径（如 /data/checkpoints.db）
        backup_dir: 备份目录（如 /data/backups）
        keep_days: 保留最近 N 天的备份，更早的自动删除

    Returns:
        备份文件路径

    Raises:
        FileNotFoundError: 源数据库不存在
        sqlite3.Error: 备份过程中数据库错误
    """
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"源数据库不存在: {db_path}")

    backup_path_root = Path(backup_dir)
    backup_path_root.mkdir(parents=True, exist_ok=True)

    # 步骤 1：WAL checkpoint，将 WAL 日志合并到主库
    # TRUNCATE 模式：合并后截断 WAL 文件，确保备份主库文件即可获得完整数据
    conn = sqlite3.connect(str(src))
    try:
        cursor = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result = cursor.fetchone()
        # result: (busy, log, checkpointed)
        # busy=0 表示成功，非 0 表示有活跃写入阻止 checkpoint
        if result and result[0] != 0:
            print(f"警告: WAL checkpoint 部分失败（busy={result[0]}），备份可能不含最新 WAL 日志",
                  file=sys.stderr)
    finally:
        conn.close()

    # 步骤 2：使用 Online Backup API 在线复制
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_file = backup_path_root / f"{src.stem}_backup_{timestamp}.db"
    tmp_file = backup_file.with_suffix(".db.tmp")

    # 先复制到 .tmp（原子性：中途崩溃不会产生半成品备份文件）
    src_conn = sqlite3.connect(str(src))
    dst_conn = sqlite3.connect(str(tmp_file))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    # 原子 rename 到最终文件名
    tmp_file.replace(backup_file)
    print(f"备份完成: {backup_file} ({backup_file.stat().st_size} bytes)")

    # 步骤 3：清理过期备份
    if keep_days > 0:
        cutoff = time.time() - keep_days * 86400
        removed = 0
        for old in backup_path_root.glob(f"{src.stem}_backup_*.db"):
            if old.stat().st_mtime < cutoff:
                old.unlink()
                removed += 1
        if removed > 0:
            print(f"清理过期备份: {removed} 个（>{keep_days} 天）")

    return str(backup_file)


def main():
    parser = argparse.ArgumentParser(description="SQLite 数据库备份工具")
    parser.add_argument(
        "--db",
        default=os.environ.get("SQLITE_CHECKPOINT_PATH", "checkpoints.db"),
        help="源数据库路径（默认: 环境变量 SQLITE_CHECKPOINT_PATH 或 checkpoints.db）",
    )
    parser.add_argument(
        "--backup-dir",
        default=os.environ.get("SQLITE_BACKUP_DIR", "./backups"),
        help="备份目录（默认: 环境变量 SQLITE_BACKUP_DIR 或 ./backups）",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=int(os.environ.get("SQLITE_BACKUP_KEEP_DAYS", "7")),
        help="保留最近 N 天的备份（默认: 7）",
    )
    args = parser.parse_args()

    try:
        backup_sqlite(args.db, args.backup_dir, args.keep)
    except Exception as e:
        print(f"备份失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

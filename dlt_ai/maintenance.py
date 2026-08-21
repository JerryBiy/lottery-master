from __future__ import annotations

import argparse
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

from .webapp import HISTORY_PATH, MODEL_DIR, RECOMMENDATION_CACHE_PATH, connect, init_db


def create_backup(output_dir: str | Path, include_models: bool = False) -> Path:
    init_db()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = output_dir / f"dlt_backend_{stamp}.zip"

    with tempfile.TemporaryDirectory() as temporary_dir:
        database_copy = Path(temporary_dir) / "app.db"
        target = sqlite3.connect(database_copy)
        try:
            with connect() as source:
                source.backup(target)
        finally:
            target.close()

        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(database_copy, "data/app.db")
            if HISTORY_PATH.exists():
                archive.write(HISTORY_PATH, "data/all_history.csv")
            if RECOMMENDATION_CACHE_PATH.exists():
                archive.write(RECOMMENDATION_CACHE_PATH, "reports/recommendation_cache.json")
            if include_models:
                if MODEL_DIR.exists():
                    for path in MODEL_DIR.rglob("*.joblib"):
                        archive.write(path, path.relative_to(MODEL_DIR.parent))
    return archive_path


def prune_operational_records() -> dict[str, int]:
    init_db()
    with connect() as db:
        refresh_deleted = db.execute(
            """
            delete from refresh_log
            where id not in (select id from refresh_log order by id desc limit 500)
            """
        ).rowcount
        jobs_deleted = db.execute(
            """
            delete from lab_jobs
            where status = 'failed'
              and datetime(updated_at) < datetime('now', '-30 days')
            """
        ).rowcount
        leases_deleted = db.execute(
            "delete from service_leases where datetime(expires_at) < datetime('now', '-1 day')"
        ).rowcount
    return {
        "refresh_logs": refresh_deleted,
        "failed_jobs": jobs_deleted,
        "expired_leases": leases_deleted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Daletou backend maintenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup_parser = subparsers.add_parser("backup")
    backup_parser.add_argument("--out", default="backups")
    backup_parser.add_argument("--include-models", action="store_true")
    subparsers.add_parser("prune")
    args = parser.parse_args()

    if args.command == "backup":
        print(create_backup(args.out, include_models=args.include_models))
    else:
        print(prune_operational_records())


if __name__ == "__main__":
    main()

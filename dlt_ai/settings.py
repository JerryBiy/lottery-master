from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    environment: str = os.getenv("DLT_ENV", "development")
    host: str = os.getenv("DLT_HOST", "127.0.0.1")
    port: int = int(os.getenv("DLT_PORT", "5000"))
    threads: int = int(os.getenv("DLT_THREADS", "8"))
    scheduler_enabled: bool = _bool_env("DLT_SCHEDULER_ENABLED", True)
    debug: bool = _bool_env("DLT_DEBUG", False)
    admin_api_key: str = os.getenv("DLT_ADMIN_API_KEY", "")
    wechat_app_id: str = os.getenv("WECHAT_APP_ID", "")
    wechat_app_secret: str = os.getenv("WECHAT_APP_SECRET", "")
    session_days: int = max(1, int(os.getenv("DLT_SESSION_DAYS", "30")))
    max_content_length: int = int(os.getenv("DLT_MAX_CONTENT_LENGTH", str(1024 * 1024)))
    job_workers: int = max(1, int(os.getenv("DLT_JOB_WORKERS", "1")))
    job_queue_limit: int = max(1, int(os.getenv("DLT_JOB_QUEUE_LIMIT", "8")))
    history_path: Path = Path(os.getenv("DLT_HISTORY_PATH", str(ROOT / "data" / "all_history.csv")))
    database_path: Path = Path(os.getenv("DLT_DATABASE_PATH", str(ROOT / "data" / "app.db")))
    log_path: Path = Path(os.getenv("DLT_LOG_PATH", str(ROOT / "reports" / "backend.log")))

    @property
    def production(self) -> bool:
        return self.environment.lower() == "production"


SETTINGS = Settings()

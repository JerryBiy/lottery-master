from __future__ import annotations

import json
import hashlib
import hmac
import importlib.util
import logging
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import requests
from flask import Flask, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from .config import BACK_NUMBERS, BACK_PICK, FRONT_NUMBERS, FRONT_PICK
from .data import fetch_official_history, fetch_recent_official_history, load_history, save_history
from .evaluate import evaluate_ticket, holdout_evaluate, recommend_models, summarize_results
from .models import FEATURE_GROUPS, MODEL_NAMES, load_model, predict_probabilities, save_model, train_model_bundle
from .optimize import generate_tickets
from .product import (
    distribution_statistics,
    draw_to_product_dict,
    generate_random_tickets,
    home_payload,
    number_statistics,
    random_candidate_counts,
)
from .settings import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
HISTORY_PATH = SETTINGS.history_path
DB_PATH = SETTINGS.database_path
RECOMMENDATION_CACHE_PATH = ROOT / "reports" / "recommendation_cache.json"
REFRESH_COOLDOWN = timedelta(minutes=10)
_REFRESH_LOCK = threading.Lock()
_LAB_COMPUTE_LOCK = threading.Lock()
_SCHEDULER_STARTED = False
_JOB_EXECUTOR = ThreadPoolExecutor(max_workers=SETTINGS.job_workers, thread_name_prefix="dlt-job")
_JOB_SLOTS = threading.BoundedSemaphore(SETTINGS.job_workers + SETTINGS.job_queue_limit)
_CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
_HISTORY_CACHE_LOCK = threading.Lock()
_HISTORY_CACHE: tuple[tuple[str, int, int], pd.DataFrame] | None = None
_PRODUCT_CACHE_LOCK = threading.Lock()
_PRODUCT_CACHE: OrderedDict[tuple[str, int, tuple[str, int, int]], Any] = OrderedDict()
LOGGER = logging.getLogger("dlt_ai.backend")


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _configure_logging() -> None:
    if LOGGER.handlers:
        return
    SETTINGS.log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        SETTINGS.log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

LAB_MODEL_META = {
    "logistic": ("Logistic", "快速、稳定的概率基线", "fast"),
    "hist_gradient_boosting": ("直方图梯度提升", "非线性关系，速度与表现均衡", "fast"),
    "gradient_boosting": ("梯度提升", "经典树提升模型", "medium"),
    "random_forest": ("随机森林", "多棵决策树集成，训练较慢", "slow"),
    "extra_trees": ("极端随机树", "更强随机性的树集成", "slow"),
    "mlp": ("MLP 神经网络", "表格特征神经网络", "medium"),
    "xgboost": ("XGBoost", "需安装可选依赖", "slow"),
    "lightgbm": ("LightGBM", "需安装可选依赖", "medium"),
}


def create_app(start_scheduler: bool = False) -> Flask:
    _configure_logging()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = SETTINGS.max_content_length
    init_db()
    if start_scheduler:
        _recover_interrupted_jobs()
        _start_refresh_scheduler()

    @app.errorhandler(Exception)
    def handle_error(exc: Exception):
        if isinstance(exc, HTTPException):
            status = int(exc.code or 500)
            message = exc.description
        elif isinstance(exc, APIError):
            status = exc.status_code
            message = str(exc)
        elif isinstance(exc, (ValueError, TypeError)):
            status = 400
            message = str(exc)
        else:
            status = 500
            message = "服务器内部错误"
            LOGGER.exception("request_failed request_id=%s path=%s", getattr(g, "request_id", "-"), request.path)
        return jsonify({"error": message, "request_id": getattr(g, "request_id", None)}), status

    @app.before_request
    def identify_request():
        g.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        g.started_at = time.perf_counter()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.after_request
    def add_api_headers(response):
        response.headers["X-Request-Id"] = getattr(g, "request_id", "")
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "no-referrer"
        elapsed_ms = round((time.perf_counter() - getattr(g, "started_at", time.perf_counter())) * 1000, 1)
        LOGGER.info(
            "request id=%s method=%s path=%s status=%s elapsed_ms=%s",
            getattr(g, "request_id", "-"),
            request.method,
            request.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.get("/api/v1/health")
    def api_v1_health():
        return jsonify({"ok": True, "service": "dlt-ai", "time": _now()})

    @app.get("/api/v1/ready")
    def api_v1_ready():
        checks: dict[str, Any] = {"database": False, "history": False}
        try:
            with connect() as db:
                db.execute("select 1").fetchone()
            checks["database"] = True
        except Exception:
            pass
        try:
            history = _safe_history(required=True)
            checks["history"] = bool(len(history))
            checks["latest_issue"] = str(history.iloc[-1]["issue"]) if len(history) else None
            checks["draw_count"] = int(len(history))
        except Exception:
            pass
        ready = bool(checks["database"] and checks["history"])
        return jsonify({"ok": ready, "checks": checks, "time": _now()}), 200 if ready else 503

    @app.post("/api/v1/auth/wechat")
    def api_v1_auth_wechat():
        if not SETTINGS.wechat_app_id or not SETTINGS.wechat_app_secret:
            raise APIError("微信登录尚未在服务器配置", 503)
        payload = request.get_json(force=True)
        code = str(payload.get("code", "")).strip()
        if not 3 <= len(code) <= 256:
            raise APIError("微信登录凭证无效")
        response = requests.get(
            "https://api.weixin.qq.com/sns/jscode2session",
            params={
                "appid": SETTINGS.wechat_app_id,
                "secret": SETTINGS.wechat_app_secret,
                "js_code": code,
                "grant_type": "authorization_code",
            },
            timeout=(5, 10),
        )
        response.raise_for_status()
        auth = response.json()
        openid = str(auth.get("openid", ""))
        if not openid:
            LOGGER.warning("wechat_login_rejected errcode=%s", auth.get("errcode"))
            raise APIError("微信登录失败，请稍后重试", 401)
        owner_key = "wechat:" + hashlib.sha256(openid.encode()).hexdigest()
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires_at = (datetime.now() + timedelta(days=SETTINGS.session_days)).isoformat(timespec="seconds")
        with connect() as db:
            db.execute(
                """
                insert into users(owner_key, openid, created_at, last_login_at)
                values (?, ?, ?, ?)
                on conflict(owner_key) do update set last_login_at = excluded.last_login_at
                """,
                (owner_key, openid, _now(), _now()),
            )
            db.execute("delete from auth_sessions where owner_key = ? or datetime(expires_at) <= datetime('now')", (owner_key,))
            db.execute(
                """
                insert into auth_sessions(token_hash, owner_key, created_at, expires_at)
                values (?, ?, ?, ?)
                """,
                (token_hash, owner_key, _now(), expires_at),
            )
        return jsonify({"token": token, "expires_at": expires_at})

    @app.get("/api/v1/home")
    def api_v1_home():
        history = _safe_history(required=True)
        payload = dict(_cached_product_payload("home", 0, history, lambda: home_payload(history)))
        payload["last_refresh"] = _last_refresh()
        return jsonify(payload)

    @app.get("/api/v1/draws")
    def api_v1_draws():
        history = _safe_history(required=True)
        query = str(request.args.get("q", "")).strip()
        limit = min(max(int(request.args.get("limit", 20)), 1), 100)
        offset = max(int(request.args.get("offset", 0)), 0)
        if query:
            history = history[history["issue"].astype(str).str.contains(query, regex=False)]
        ordered = history.iloc[::-1]
        page = ordered.iloc[offset : offset + limit]
        return jsonify(
            {
                "total": int(len(history)),
                "offset": offset,
                "has_more": offset + limit < len(history),
                "rows": [draw_to_product_dict(row) for _, row in page.iterrows()],
            }
        )

    @app.get("/api/v1/statistics/numbers")
    def api_v1_number_statistics():
        history = _safe_history(required=True)
        window = int(request.args.get("window", 100))
        return jsonify(
            _cached_product_payload(
                "numbers",
                window,
                history,
                lambda: number_statistics(history, window),
            )
        )

    @app.get("/api/v1/statistics/distributions")
    def api_v1_distribution_statistics():
        history = _safe_history(required=True)
        window = int(request.args.get("window", 100))
        return jsonify(
            _cached_product_payload(
                "distributions",
                window,
                history,
                lambda: distribution_statistics(history, window),
            )
        )

    @app.post("/api/v1/random")
    def api_v1_random():
        payload = request.get_json(silent=True) or {}
        count = min(max(int(payload.get("count", 5)), 1), 20)
        constraints = payload.get("constraints") or {}
        if not isinstance(constraints, dict):
            raise ValueError("自定义条件格式不正确")
        history = _safe_history(required=True)
        tickets = generate_random_tickets(count, constraints=constraints)
        return jsonify(
            {
                "tickets": tickets,
                "candidate_counts": random_candidate_counts(constraints),
                "customized": bool(constraints),
                "latest_issue": str(history.iloc[-1]["issue"]),
                "statement": "号码由等概率随机过程生成，不代表预测或推荐。",
            }
        )

    @app.post("/api/v1/favorites/evaluate")
    def api_v1_favorites_evaluate():
        payload = request.get_json(silent=True) or {}
        groups = payload.get("groups")
        if groups is None:
            groups = _favorite_groups(_owner_key(), limit=100)
        if not isinstance(groups, list):
            raise ValueError("收藏数据格式不正确")
        history = _safe_history(required=True)
        return jsonify({"evaluations": _evaluate_favorite_groups(history, groups[:100])})

    @app.get("/api/v1/favorites")
    def api_v1_favorites():
        limit = min(max(int(request.args.get("limit", 50)), 1), 100)
        return jsonify({"groups": _favorite_groups(_owner_key(), limit=limit)})

    @app.post("/api/v1/favorites")
    def api_v1_favorites_save():
        payload = request.get_json(force=True)
        group = _validate_favorite_group(payload)
        _save_favorite_group(_owner_key(), group)
        return jsonify({"ok": True, "group": group}), 201

    @app.delete("/api/v1/favorites/<favorite_id>")
    def api_v1_favorites_delete(favorite_id: str):
        deleted = _delete_favorite_group(_owner_key(), favorite_id)
        if not deleted:
            raise APIError("收藏记录不存在", 404)
        return jsonify({"ok": True, "id": favorite_id})

    @app.post("/api/v1/refresh")
    def api_v1_refresh():
        payload = request.get_json(silent=True) or {}
        force = bool(payload.get("force", False))
        if force:
            _require_admin()
        result = refresh_history(force=force)
        result["evaluated_predictions"] = evaluate_saved_predictions()
        return jsonify(result)

    @app.get("/api/v1/about")
    def api_v1_about():
        return jsonify(
            {
                "product_name": "大乐透数据观察",
                "positioning": "历史开奖数据分析与概率教育工具",
                "data_source": "中国体育彩票公开开奖数据",
                "service_boundary": [
                    "不销售、代购或合买任何彩票",
                    "不承诺中奖，不提供所谓必中号码",
                    "不接入售彩平台或投注平台",
                    "所有统计仅用于历史数据观察和概率教育",
                ],
            }
        )

    @app.get("/api/v1/lab/overview")
    def api_v1_lab_overview():
        owner_key = _owner_key()
        history = _safe_history(required=True)
        evaluate_saved_predictions()
        cached = _load_recommendation_cache()
        recommendation = _lab_default_recommendation(cached, str(history.iloc[-1]["issue"]))
        return jsonify(
            {
                "latest_issue": str(history.iloc[-1]["issue"]),
                "total_draws": int(len(history)),
                "models": _lab_models(),
                "windows": [
                    {"value": 100, "label": "近100期"},
                    {"value": 300, "label": "近300期"},
                    {"value": 500, "label": "近500期"},
                    {"value": 1000, "label": "近1000期"},
                    {"value": 0, "label": "全部历史"},
                ],
                "recommendation": recommendation,
                "metrics": _lab_metrics(owner_key),
                "predictions": _lab_predictions(limit=8, owner_key=owner_key),
                "jobs": _lab_jobs(limit=8, owner_key=owner_key),
                "generation_history": _lab_generation_history(limit=30, owner_key=owner_key),
                "feature_groups": [
                    {"id": "trend", "label": "近期走势", "short": "近期", "items": "滚动频次、出现率和EMA趋势"},
                    {"id": "heat", "label": "冷热遗漏", "short": "冷热", "items": "当前遗漏、冷热排名和标准分"},
                    {"id": "cycle", "label": "间隔周期", "short": "周期", "items": "平均间隔、偏离程度和连续出现"},
                    {"id": "properties", "label": "号码属性", "short": "属性", "items": "奇偶、大小、质数、尾数和三区"},
                    {"id": "context", "label": "上期结构", "short": "结构", "items": "上期和值、跨度、奇偶和大小"},
                    {"id": "relations", "label": "号码关系", "short": "关系", "items": "邻号、镜像号和上期关联"},
                ],
                "objectives": [
                    {"id": "balanced", "label": "综合均衡", "description": "兼顾模型评分与常见结构"},
                    {"id": "score", "label": "评分优先", "description": "更集中选择模型高分号码"},
                    {"id": "structure", "label": "结构优先", "description": "贴近历史和值与分区分布"},
                    {"id": "coverage", "label": "多注覆盖", "description": "降低多注之间的号码重复"},
                ],
                "presets": [
                    {"id": "conservative", "label": "稳健", "description": "约束更强，降低过拟合"},
                    {"id": "standard", "label": "标准", "description": "复杂度与速度均衡"},
                    {"id": "exploratory", "label": "探索", "description": "模型更复杂，训练更慢"},
                ],
                "statement": "模型只输出基于历史特征的相对评分，结果不构成中奖承诺。",
            }
        )

    @app.post("/api/v1/lab/jobs/generate")
    def api_v1_lab_generate():
        payload = request.get_json(force=True)
        model = _validate_lab_model(payload.get("model", "logistic"))
        params = {
            "_owner_key": _owner_key(),
            "model": model,
            "train_window": _lab_window(payload.get("trainWindow", 300)),
            "tickets": min(max(int(payload.get("tickets", 5)), 1), 20),
            "candidates": min(max(int(payload.get("candidates", 3000)), 200), 50_000),
            "source_mode": _lab_source_mode(payload.get("sourceMode")),
            **_lab_custom_params(payload),
            "save": bool(payload.get("save", True)),
        }
        job_id = _create_lab_job("generate", params, "等待开始", params["_owner_key"])
        if not _start_lab_job(job_id, _run_lab_generate, params):
            raise APIError("当前模型任务较多，请稍后再试", 429)
        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.post("/api/v1/lab/jobs/backtest")
    def api_v1_lab_backtest():
        payload = request.get_json(force=True)
        history = _safe_history(required=True)
        test_draws = min(max(int(payload.get("testDraws", 20)), 5), 60)
        cutoff_index = max(40, len(history) - test_draws - 1)
        params = {
            "_owner_key": _owner_key(),
            "model": _validate_lab_model(payload.get("model", "logistic")),
            "train_window": _lab_window(payload.get("trainWindow", 300)),
            "tickets": min(max(int(payload.get("tickets", 3)), 1), 10),
            "candidates": min(max(int(payload.get("candidates", 500)), 100), 5000),
            "test_draws": test_draws,
            "cutoff_issue": str(history.iloc[cutoff_index]["issue"]),
            **_lab_custom_params(payload),
        }
        job_id = _create_lab_job("backtest", params, "等待开始", params["_owner_key"])
        if not _start_lab_job(job_id, _run_lab_backtest, params):
            raise APIError("当前模型任务较多，请稍后再试", 429)
        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.post("/api/v1/lab/jobs/recommend")
    def api_v1_lab_recommend():
        payload = request.get_json(silent=True) or {}
        requested_models = payload.get("models") or ["logistic", "hist_gradient_boosting"]
        models = [_validate_lab_model(model) for model in requested_models][:4]
        params = {
            "_owner_key": _owner_key(),
            "models": models,
            "windows": [_lab_window(value) for value in (payload.get("windows") or [300, 500])][:4],
            "test_draws": min(max(int(payload.get("testDraws", 12)), 5), 30),
            "tickets": 2,
            "candidates": min(max(int(payload.get("candidates", 300)), 100), 1000),
        }
        job_id = _create_lab_job("recommend", params, "等待开始", params["_owner_key"])
        if not _start_lab_job(job_id, _run_lab_recommend, params):
            raise APIError("当前模型任务较多，请稍后再试", 429)
        return jsonify({"job_id": job_id, "status": "queued"}), 202

    @app.get("/api/v1/lab/jobs/<job_id>")
    def api_v1_lab_job(job_id: str):
        job = _lab_job(job_id, _owner_key())
        if not job:
            return jsonify({"error": "任务不存在"}), 404
        return jsonify(job)

    @app.post("/api/v1/lab/history/<job_id>/delete")
    def api_v1_lab_history_delete(job_id: str):
        owner_key = _owner_key()
        with connect() as db:
            row = db.execute(
                "select job_type, status from lab_jobs where id = ? and owner_key = ?",
                (job_id, owner_key),
            ).fetchone()
            if not row or row["job_type"] != "generate" or row["status"] != "completed":
                return jsonify({"error": "生成历史不存在"}), 404
            db.execute("delete from lab_jobs where id = ? and owner_key = ?", (job_id, owner_key))
        return jsonify({"ok": True, "id": job_id})

    @app.post("/api/refresh")
    def api_refresh():
        _require_admin()
        result = refresh_history()
        evaluated = evaluate_saved_predictions()
        result["evaluated_predictions"] = evaluated
        return jsonify(result)

    @app.get("/api/summary")
    def api_summary():
        history = _safe_history()
        latest = _draw_to_dict(history.iloc[-1]) if len(history) else None
        return jsonify(
            {
                "draw_count": int(len(history)),
                "latest": latest,
                "models": _model_status(history),
                "prediction_count": _scalar("select count(*) from predictions"),
                "pending_count": _scalar("select count(*) from predictions where status = 'pending'"),
            }
        )

    @app.get("/api/history")
    def api_history():
        history = _safe_history()
        query = str(request.args.get("q", "")).strip()
        limit = min(int(request.args.get("limit", 30)), 200)
        offset = max(int(request.args.get("offset", 0)), 0)
        if query:
            history = history[history["issue"].astype(str).str.contains(query, regex=False)]
        total = len(history)
        page = history.sort_values("issue", key=lambda s: pd.to_numeric(s, errors="coerce"), ascending=False).iloc[offset : offset + limit]
        return jsonify({"total": int(total), "rows": [_draw_to_dict(row) for _, row in page.iterrows()]})

    @app.post("/api/train")
    def api_train():
        _require_admin()
        payload = request.get_json(force=True)
        model_name = payload.get("model", "logistic")
        history = _safe_history(required=True)
        train_window = _resolve_train_window(history, payload)
        bundle = train_model_bundle(history, model_name=model_name, train_window_draws=train_window)
        path = _model_path(model_name, train_window)
        save_model(bundle, path)
        return jsonify({"ok": True, "model": model_name, "path": str(path), "summary": _model_summary(bundle)})

    @app.post("/api/predict")
    def api_predict():
        _require_admin()
        payload = request.get_json(force=True)
        mode = payload.get("mode", "model")
        model_name = payload.get("model", "logistic")
        count = min(max(int(payload.get("tickets", 10)), 1), 200)
        candidates = min(max(int(payload.get("candidates", 20_000)), 100), 1_000_000)
        save = bool(payload.get("save", True))
        auto_train = bool(payload.get("autoTrain", True))

        history = _safe_history(required=True)
        train_window = _resolve_train_window(history, payload)
        trained_until_issue = str(history.iloc[-1]["issue"])
        if mode == "random":
            tickets = _random_ticket_dicts(count)
            model_used = "random"
        else:
            bundle = _load_or_train_bundle(history, model_name, train_window, auto_train=auto_train)
            probabilities = predict_probabilities(bundle, history)
            tickets = [ticket.as_dict() for ticket in generate_tickets(probabilities, history, candidates=candidates, top_k=count)]
            model_used = model_name

        prediction_id = None
        if save:
            prediction_id = save_prediction(
                mode=mode,
                model_name=model_used,
                trained_until_issue=trained_until_issue,
                tickets=tickets,
                owner_key=_owner_key(),
            )
            evaluate_saved_predictions()
        return jsonify(
            {
                "prediction_id": prediction_id,
                "mode": mode,
                "model": model_used,
                "train_window_draws": train_window,
                "trained_until_issue": trained_until_issue,
                "tickets": tickets,
            }
        )

    @app.route("/api/evaluate", methods=["GET", "POST"])
    def api_evaluate():
        return jsonify({"evaluated": evaluate_saved_predictions()})

    @app.get("/api/predictions")
    def api_predictions():
        limit = min(int(request.args.get("limit", 50)), 200)
        with connect() as db:
            rows = db.execute(
                """
                select id, created_at, mode, model_name, trained_until_issue, target_issue,
                       tickets_json, status, evaluated_at, best_front_hits, best_back_hits, best_prize
                from predictions
                where owner_key = ?
                order by id desc
                limit ?
                """,
                (_owner_key(), limit),
            ).fetchall()
        return jsonify([_prediction_row(row) for row in rows])

    @app.get("/api/metrics")
    def api_metrics():
        with connect() as db:
            rows = db.execute(
                """
                select p.mode, p.model_name, r.front_hits, r.back_hits, r.prize_level
                from prediction_results r
                join predictions p on p.id = r.prediction_id
                where p.owner_key = ?
                """
                ,
                (_owner_key(),),
            ).fetchall()
        if not rows:
            return jsonify([])
        frame = pd.DataFrame([dict(row) for row in rows])
        out = []
        for (mode, model), group in frame.groupby(["mode", "model_name"]):
            out.append(
                {
                    "mode": mode,
                    "model": model,
                    "tickets": int(len(group)),
                    "avg_front_hits": float(group["front_hits"].mean()),
                    "avg_back_hits": float(group["back_hits"].mean()),
                    "prize_rate": float(group["prize_level"].notna().mean()),
                    "best_prize": int(group["prize_level"].dropna().min()) if group["prize_level"].notna().any() else None,
                }
            )
        return jsonify(out)

    @app.post("/api/holdout")
    def api_holdout():
        _require_admin()
        payload = request.get_json(force=True)
        model_name = payload.get("model", "logistic")
        train_until_issue = str(payload.get("trainUntilIssue", "")).strip()
        if not train_until_issue:
            raise ValueError("trainUntilIssue is required")
        tickets = min(max(int(payload.get("tickets", 5)), 1), 50)
        candidates = min(max(int(payload.get("candidates", 1_000)), 100), 100_000)
        max_test_draws = payload.get("maxTestDraws")
        max_test_draws = None if max_test_draws in (None, "", 0) else min(max(int(max_test_draws), 1), 500)
        history = _safe_history(required=True)
        train_window = _resolve_train_window(history, payload)
        results = holdout_evaluate(
            history,
            model_name=model_name,
            train_until_issue=train_until_issue,
            tickets_per_draw=tickets,
            candidates=candidates,
            max_test_draws=max_test_draws,
            train_window_draws=train_window,
        )
        summary = summarize_results(results)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detail_path = ROOT / "reports" / f"holdout_{model_name}_{train_until_issue}_{stamp}.csv"
        summary_path = ROOT / "reports" / f"holdout_{model_name}_{train_until_issue}_{stamp}_summary.csv"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(detail_path, index=False, encoding="utf-8")
        summary.to_csv(summary_path, index=False, encoding="utf-8")
        return jsonify(
            {
                "rows": summary.replace({np.nan: None}).to_dict(orient="records"),
                "detail_path": str(detail_path),
                "summary_path": str(summary_path),
                "test_draws": int(results["issue"].nunique()),
                "tickets_per_strategy": int(results.groupby("strategy").size().max()),
            }
        )

    @app.post("/api/recommend")
    def api_recommend():
        _require_admin()
        payload = request.get_json(force=True)
        models = payload.get("models") or ["logistic", "hist_gradient_boosting", "gradient_boosting"]
        windows = [_parse_window_value(item) for item in (payload.get("windows") or [300, 500, 1000, "all"])]
        train_until_issue = str(payload.get("trainUntilIssue", "")).strip()
        if not train_until_issue:
            raise ValueError("trainUntilIssue is required")
        tickets = min(max(int(payload.get("tickets", 3)), 1), 20)
        candidates = min(max(int(payload.get("candidates", 500)), 100), 20_000)
        max_test_draws = min(max(int(payload.get("maxTestDraws", 30)), 1), 200)
        history = _safe_history(required=True)
        recommendations = recommend_models(
            history,
            model_names=models,
            train_windows=windows,
            train_until_issue=train_until_issue,
            tickets_per_draw=tickets,
            candidates=candidates,
            max_test_draws=max_test_draws,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "reports" / f"recommendations_{stamp}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(out_path, index=False, encoding="utf-8")
        return jsonify({"rows": recommendations.replace({np.nan: None}).to_dict(orient="records"), "path": str(out_path)})

    @app.get("/api/recommend/default")
    def api_default_recommend():
        history = _safe_history(required=True)
        latest_issue = str(history.iloc[-1]["issue"])
        cached = _load_recommendation_cache()
        if cached:
            return jsonify({**cached, "cached": True, "stale": cached.get("latest_issue") != latest_issue})

        payload = {
            "models": ["logistic"],
            "windows": [300, None],
            "train_until_issue": _default_recommend_cutoff(history),
            "tickets": 1,
            "candidates": 50,
            "max_test_draws": 3,
        }
        recommendations = recommend_models(
            history,
            model_names=payload["models"],
            train_windows=payload["windows"],
            train_until_issue=payload["train_until_issue"],
            tickets_per_draw=payload["tickets"],
            candidates=payload["candidates"],
            max_test_draws=payload["max_test_draws"],
        )
        data = {
            "latest_issue": latest_issue,
            "created_at": _now(),
            "cached": False,
            "stale": False,
            "params": payload,
            "rows": recommendations.replace({np.nan: None}).to_dict(orient="records"),
        }
        data = _json_clean(data)
        _save_recommendation_cache(data)
        return jsonify(data)

    return app


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as db:
        db.execute("pragma journal_mode = wal")
        db.execute(
            """
            create table if not exists predictions (
                id integer primary key autoincrement,
                created_at text not null,
                mode text not null,
                model_name text not null,
                trained_until_issue text not null,
                target_issue text,
                tickets_json text not null,
                status text not null,
                evaluated_at text,
                best_front_hits integer,
                best_back_hits integer,
                best_prize integer
            )
            """
        )
        _ensure_column(db, "predictions", "owner_key", "text not null default 'local-development'")
        db.execute(
            """
            create table if not exists prediction_results (
                id integer primary key autoincrement,
                prediction_id integer not null,
                rank integer not null,
                front text not null,
                back text not null,
                front_hits integer not null,
                back_hits integer not null,
                prize_level integer,
                foreign key(prediction_id) references predictions(id)
            )
            """
        )
        db.execute(
            """
            create table if not exists refresh_log (
                id integer primary key autoincrement,
                created_at text not null,
                ok integer not null,
                message text not null,
                before_count integer,
                after_count integer,
                latest_issue text
            )
            """
        )
        db.execute(
            """
            create table if not exists lab_jobs (
                id text primary key,
                job_type text not null,
                status text not null,
                progress integer not null,
                message text not null,
                params_json text not null,
                result_json text,
                error text,
                created_at text not null,
                updated_at text not null
            )
            """
        )
        _ensure_column(db, "lab_jobs", "owner_key", "text not null default 'local-development'")
        db.execute(
            """
            create table if not exists users (
                owner_key text primary key,
                openid text not null unique,
                created_at text not null,
                last_login_at text not null
            )
            """
        )
        db.execute(
            """
            create table if not exists auth_sessions (
                token_hash text primary key,
                owner_key text not null,
                created_at text not null,
                expires_at text not null,
                foreign key(owner_key) references users(owner_key) on delete cascade
            )
            """
        )
        db.execute(
            """
            create table if not exists favorite_groups (
                owner_key text not null,
                id text not null,
                created_at text not null,
                updated_at text not null,
                source text not null,
                generated_after_issue text,
                payload_json text not null,
                primary key (owner_key, id)
            )
            """
        )
        db.execute(
            """
            create table if not exists service_leases (
                name text primary key,
                owner text not null,
                expires_at text not null
            )
            """
        )
        db.execute("create index if not exists idx_predictions_owner_created on predictions(owner_key, created_at desc)")
        db.execute("create index if not exists idx_predictions_status on predictions(status)")
        db.execute("create index if not exists idx_prediction_results_prediction on prediction_results(prediction_id)")
        db.execute("create index if not exists idx_lab_jobs_owner_created on lab_jobs(owner_key, created_at desc)")
        db.execute("create index if not exists idx_lab_jobs_status on lab_jobs(status)")
        db.execute("create index if not exists idx_refresh_log_created on refresh_log(created_at desc)")
        db.execute("create index if not exists idx_favorites_owner_created on favorite_groups(owner_key, created_at desc)")
        db.execute("create index if not exists idx_auth_sessions_owner on auth_sessions(owner_key)")
        db.execute("create index if not exists idx_auth_sessions_expiry on auth_sessions(expires_at)")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma foreign_keys = on")
    conn.execute("pragma synchronous = normal")
    conn.execute("pragma busy_timeout = 30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_column(db: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in db.execute(f"pragma table_info({table})")}
    if column not in columns:
        db.execute(f"alter table {table} add column {column} {definition}")


def refresh_history(force: bool = False) -> dict[str, Any]:
    if not _REFRESH_LOCK.acquire(blocking=False):
        before = _safe_history()
        return {
            "ok": True,
            "message": "数据刷新正在后台进行",
            "before_count": int(len(before)),
            "after_count": int(len(before)),
            "latest_issue": str(before.iloc[-1]["issue"]) if len(before) else None,
            "cached": True,
            "checked_at": _now(),
        }
    try:
        return _refresh_history_locked(force)
    finally:
        _REFRESH_LOCK.release()


def _refresh_history_locked(force: bool = False) -> dict[str, Any]:
    before = _safe_history()
    before_count = len(before)
    last_refresh = _last_refresh()
    if (
        not force
        and last_refresh
        and bool(last_refresh["ok"])
        and datetime.now() - datetime.fromisoformat(last_refresh["created_at"]) < REFRESH_COOLDOWN
    ):
        return {
            "ok": bool(last_refresh["ok"]),
            "message": "数据刚刚检查过，无需重复刷新",
            "before_count": int(before_count),
            "after_count": int(before_count),
            "latest_issue": str(before.iloc[-1]["issue"]) if len(before) else None,
            "cached": True,
            "checked_at": last_refresh["created_at"],
        }
    try:
        fetched = fetch_recent_official_history(page_size=100) if before_count else fetch_official_history()
        source = fetched.attrs.get("source", "中国体育彩票")
        _validate_fetched_history(before, fetched)
        merged = _merge_history(before, fetched)
        save_history(merged, HISTORY_PATH)
        after = load_history(HISTORY_PATH)
        result = {
            "ok": True,
            "message": f"从{source}更新 {len(after) - before_count:+d} 期",
            "before_count": int(before_count),
            "after_count": int(len(after)),
            "latest_issue": str(after.iloc[-1]["issue"]) if len(after) else None,
            "cached": False,
            "checked_at": _now(),
            "source": source,
        }
    except Exception as exc:
        latest = str(before.iloc[-1]["issue"]) if len(before) else None
        result = {
            "ok": False,
            "message": f"auto refresh failed: {exc}",
            "before_count": int(before_count),
            "after_count": int(before_count),
            "latest_issue": latest,
            "cached": False,
            "checked_at": _now(),
        }
    _log_refresh(result)
    return result


def _start_refresh_scheduler() -> None:
    global _SCHEDULER_STARTED
    if _SCHEDULER_STARTED:
        return
    _SCHEDULER_STARTED = True
    worker_id = uuid.uuid4().hex

    def worker() -> None:
        time.sleep(1)
        while True:
            now = datetime.now()
            is_draw_window = now.weekday() in {0, 2, 5} and (20 <= now.hour <= 23)
            interval = 300 if is_draw_window else 3600
            try:
                _prewarm_product_cache()
                if _acquire_service_lease("history-refresh", worker_id, 120):
                    result = refresh_history()
                    evaluated = evaluate_saved_predictions()
                    _prewarm_product_cache()
                    LOGGER.info(
                        "scheduled_refresh ok=%s latest_issue=%s evaluated=%s",
                        result.get("ok"),
                        result.get("latest_issue"),
                        evaluated,
                    )
            except Exception:
                LOGGER.exception("scheduled_refresh_failed")
            time.sleep(interval)

    threading.Thread(target=worker, name="dlt-history-refresh", daemon=True).start()


def _validate_fetched_history(before: pd.DataFrame, fetched: pd.DataFrame) -> None:
    if fetched.empty:
        raise RuntimeError("开奖数据源返回空数据")
    if fetched["issue"].duplicated().any():
        raise RuntimeError("开奖数据源返回重复期号")
    if before.empty:
        return
    old = before.set_index("issue")
    new = fetched.set_index("issue")
    overlap = old.index.intersection(new.index)
    number_columns = [f"front{i}" for i in range(1, 6)] + [f"back{i}" for i in range(1, 3)]
    for issue in overlap:
        old_numbers = tuple(int(old.loc[issue, column]) for column in number_columns)
        new_numbers = tuple(int(new.loc[issue, column]) for column in number_columns)
        if old_numbers != new_numbers:
            raise RuntimeError(f"数据源中第 {issue} 期号码与本地记录冲突")


def _acquire_service_lease(name: str, owner: str, ttl_seconds: int) -> bool:
    now = datetime.now()
    expires_at = (now + timedelta(seconds=ttl_seconds)).isoformat(timespec="seconds")
    with connect() as db:
        db.execute("begin immediate")
        row = db.execute("select owner, expires_at from service_leases where name = ?", (name,)).fetchone()
        if row and row["owner"] != owner and datetime.fromisoformat(row["expires_at"]) > now:
            return False
        db.execute(
            """
            insert into service_leases(name, owner, expires_at) values (?, ?, ?)
            on conflict(name) do update set owner = excluded.owner, expires_at = excluded.expires_at
            """,
            (name, owner, expires_at),
        )
    return True


def _recover_interrupted_jobs() -> None:
    with connect() as db:
        cursor = db.execute(
            """
            update lab_jobs
            set status = 'failed', progress = 100, message = '服务重启，任务已中止',
                error = 'service restarted before task completion', updated_at = ?
            where status in ('queued', 'running')
            """,
            (_now(),),
        )
    if cursor.rowcount:
        LOGGER.warning("recovered_interrupted_jobs count=%s", cursor.rowcount)


def _prewarm_product_cache() -> None:
    try:
        history = _safe_history(required=True)
        _cached_product_payload("home", 0, history, lambda: home_payload(history))
        for window in (100, 30, 50, 300):
            _cached_product_payload(
                "numbers",
                window,
                history,
                lambda window=window: number_statistics(history, window),
            )
            _cached_product_payload(
                "distributions",
                window,
                history,
                lambda window=window: distribution_statistics(history, window),
            )
    except Exception:
        LOGGER.exception("product_cache_prewarm_failed")


def evaluate_saved_predictions() -> int:
    history = _safe_history()
    if history.empty:
        return 0
    issues = history["issue"].map(_issue_key).to_numpy()
    evaluated = 0
    with connect() as db:
        rows = db.execute(
            """
            select id, trained_until_issue, tickets_json
            from predictions
            where status = 'pending'
            order by id
            """
        ).fetchall()
        for row in rows:
            trained_key = _issue_key(row["trained_until_issue"])
            target_idx = int(np.searchsorted(issues, trained_key, side="right"))
            if target_idx >= len(history):
                continue
            target = history.iloc[target_idx]
            tickets = json.loads(row["tickets_json"])
            hits = []
            db.execute("delete from prediction_results where prediction_id = ?", (row["id"],))
            for rank, ticket in enumerate(tickets, start=1):
                front = _parse_numbers(ticket["front"])
                back = _parse_numbers(ticket["back"])
                hit = evaluate_ticket(front, back, target)
                hits.append(hit)
                db.execute(
                    """
                    insert into prediction_results
                    (prediction_id, rank, front, back, front_hits, back_hits, prize_level)
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"],
                        rank,
                        ticket["front"],
                        ticket["back"],
                        hit.front_hits,
                        hit.back_hits,
                        hit.prize_level,
                    ),
                )
            best = sorted(hits, key=lambda h: (h.prize_level is None, h.prize_level or 99, -h.front_hits, -h.back_hits))[0]
            db.execute(
                """
                update predictions
                set status = 'evaluated', target_issue = ?, evaluated_at = ?,
                    best_front_hits = ?, best_back_hits = ?, best_prize = ?
                where id = ?
                """,
                (
                    str(target["issue"]),
                    _now(),
                    best.front_hits,
                    best.back_hits,
                    best.prize_level,
                    row["id"],
                ),
            )
            evaluated += 1
    return evaluated


def save_prediction(
    mode: str,
    model_name: str,
    trained_until_issue: str,
    tickets: list[dict[str, Any]],
    owner_key: str = "local-development",
) -> int:
    with connect() as db:
        cursor = db.execute(
            """
            insert into predictions
            (created_at, mode, model_name, trained_until_issue, tickets_json, status, owner_key)
            values (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (_now(), mode, model_name, trained_until_issue, json.dumps(tickets, ensure_ascii=False), owner_key),
        )
        return int(cursor.lastrowid)


def _owner_key() -> str:
    authorization = str(request.headers.get("Authorization", ""))
    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with connect() as db:
            row = db.execute(
                """
                select owner_key from auth_sessions
                where token_hash = ? and datetime(expires_at) > datetime('now')
                """,
                (token_hash,),
            ).fetchone()
        if row:
            return str(row["owner_key"])
        raise APIError("登录状态已失效，请重新登录", 401)
    if SETTINGS.production:
        raise APIError("请先登录", 401)
    value = str(request.headers.get("X-Client-Id", "")).strip()
    if not value:
        return "local-development"
    if not _CLIENT_ID_PATTERN.fullmatch(value):
        raise APIError("客户端标识格式不正确", 400)
    if not SETTINGS.production:
        _adopt_legacy_owner(value)
    return value


def _require_admin() -> None:
    configured = SETTINGS.admin_api_key
    if not configured and not SETTINGS.production:
        return
    supplied = request.headers.get("X-Admin-Key", "")
    if not configured or not hmac.compare_digest(supplied, configured):
        raise APIError("无权执行强制刷新", 403)


def _adopt_legacy_owner(owner_key: str) -> None:
    with connect() as db:
        already_present = db.execute(
            "select 1 from lab_jobs where owner_key = ? limit 1",
            (owner_key,),
        ).fetchone()
        if already_present:
            return
        db.execute(
            "update lab_jobs set owner_key = ? where owner_key = 'local-development'",
            (owner_key,),
        )
        db.execute(
            "update predictions set owner_key = ? where owner_key = 'local-development'",
            (owner_key,),
        )


def _validate_favorite_group(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise APIError("收藏数据格式不正确")
    favorite_id = str(payload.get("id") or uuid.uuid4().hex)
    if not _CLIENT_ID_PATTERN.fullmatch(favorite_id):
        raise APIError("收藏记录 ID 格式不正确")
    tickets = payload.get("tickets") or []
    if not isinstance(tickets, list) or not 1 <= len(tickets) <= 20:
        raise APIError("每组收藏必须包含 1 至 20 注号码")
    normalized_tickets = []
    for ticket in tickets:
        front = _favorite_numbers(ticket.get("front", ticket.get("front_text", "")))
        back = _favorite_numbers(ticket.get("back", ticket.get("back_text", "")))
        if (
            len(front) != FRONT_PICK
            or len(set(front)) != FRONT_PICK
            or not all(number in FRONT_NUMBERS for number in front)
            or len(back) != BACK_PICK
            or len(set(back)) != BACK_PICK
            or not all(number in BACK_NUMBERS for number in back)
        ):
            raise APIError("收藏中包含无效号码")
        normalized_tickets.append(
            {
                **ticket,
                "front": list(sorted(front)),
                "back": list(sorted(back)),
                "front_text": _format_numbers(tuple(sorted(front))),
                "back_text": _format_numbers(tuple(sorted(back))),
            }
        )
    now = str(payload.get("createdAt") or payload.get("created_at") or _now())
    return {
        **payload,
        "id": favorite_id,
        "createdAt": now,
        "source": str(payload.get("source") or "unknown")[:40],
        "generatedAfterIssue": str(payload.get("generatedAfterIssue") or "") or None,
        "tickets": normalized_tickets,
    }


def _save_favorite_group(owner_key: str, group: dict[str, Any]) -> None:
    now = _now()
    with connect() as db:
        db.execute(
            """
            insert into favorite_groups
            (owner_key, id, created_at, updated_at, source, generated_after_issue, payload_json)
            values (?, ?, ?, ?, ?, ?, ?)
            on conflict(owner_key, id) do update set
                updated_at = excluded.updated_at,
                source = excluded.source,
                generated_after_issue = excluded.generated_after_issue,
                payload_json = excluded.payload_json
            """,
            (
                owner_key,
                group["id"],
                group["createdAt"],
                now,
                group["source"],
                group["generatedAfterIssue"],
                json.dumps(group, ensure_ascii=False),
            ),
        )


def _favorite_groups(owner_key: str, limit: int) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select payload_json from favorite_groups
            where owner_key = ?
            order by created_at desc
            limit ?
            """,
            (owner_key, limit),
        ).fetchall()
    return [json.loads(row["payload_json"]) for row in rows]


def _delete_favorite_group(owner_key: str, favorite_id: str) -> bool:
    with connect() as db:
        cursor = db.execute(
            "delete from favorite_groups where owner_key = ? and id = ?",
            (owner_key, favorite_id),
        )
        return cursor.rowcount > 0


def _load_or_train_bundle(history: pd.DataFrame, model_name: str, train_window: int | None, auto_train: bool) -> dict[str, Any]:
    path = _model_path(model_name, train_window)
    if path.exists():
        bundle = load_model(path)
        if str(bundle["front"].get("last_issue")) == str(history.iloc[-1]["issue"]):
            return bundle
        if not auto_train:
            return bundle
    bundle = train_model_bundle(history, model_name=model_name, train_window_draws=train_window)
    save_model(bundle, path)
    return bundle


def _safe_history(required: bool = False) -> pd.DataFrame:
    global _HISTORY_CACHE
    if not HISTORY_PATH.exists():
        if required:
            raise FileNotFoundError(f"history file not found: {HISTORY_PATH}")
        return pd.DataFrame(columns=["issue", "date", "front1", "front2", "front3", "front4", "front5", "back1", "back2"])
    version = _history_version()
    with _HISTORY_CACHE_LOCK:
        if _HISTORY_CACHE is not None and _HISTORY_CACHE[0] == version:
            return _HISTORY_CACHE[1].copy(deep=False)
    history = load_history(HISTORY_PATH)
    with _HISTORY_CACHE_LOCK:
        _HISTORY_CACHE = (version, history)
    return history.copy(deep=False)


def _history_version() -> tuple[str, int, int]:
    stat = HISTORY_PATH.stat()
    return str(HISTORY_PATH.resolve()), stat.st_mtime_ns, stat.st_size


def _cached_product_payload(
    kind: str,
    window: int,
    history: pd.DataFrame,
    builder,
) -> Any:
    version = _history_version()
    key = (kind, int(window), version)
    with _PRODUCT_CACHE_LOCK:
        cached = _PRODUCT_CACHE.get(key)
        if cached is not None:
            _PRODUCT_CACHE.move_to_end(key)
            return cached
    value = builder()
    with _PRODUCT_CACHE_LOCK:
        _PRODUCT_CACHE[key] = value
        _PRODUCT_CACHE.move_to_end(key)
        while len(_PRODUCT_CACHE) > 16:
            _PRODUCT_CACHE.popitem(last=False)
    return value


def _merge_history(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if a.empty:
        return b
    if b.empty:
        return a
    merged = pd.concat([a, b], ignore_index=True)
    return merged.drop_duplicates("issue", keep="last").sort_values("issue", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)


def _draw_to_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "issue": str(row["issue"]),
        "date": str(pd.to_datetime(row["date"]).date()),
        "front": [int(row[f"front{i}"]) for i in range(1, 6)],
        "back": [int(row[f"back{i}"]) for i in range(1, 3)],
        "front_text": " ".join(f"{int(row[f'front{i}']):02d}" for i in range(1, 6)),
        "back_text": " ".join(f"{int(row[f'back{i}']):02d}" for i in range(1, 3)),
    }


def _model_status(history: pd.DataFrame) -> list[dict[str, Any]]:
    latest = str(history.iloc[-1]["issue"]) if len(history) else None
    out = []
    for model in ("logistic", "hist_gradient_boosting", "gradient_boosting", "random_forest", "extra_trees", "mlp"):
        path = _model_path(model)
        item = {"name": model, "exists": path.exists(), "fresh": False, "last_issue": None, "size_mb": None}
        if path.exists():
            item["size_mb"] = round(path.stat().st_size / 1024 / 1024, 2)
            try:
                bundle = load_model(path)
                item["last_issue"] = str(bundle["front"].get("last_issue"))
                item["fresh"] = item["last_issue"] == latest
            except Exception:
                item["last_issue"] = "unreadable"
        out.append(item)
    return out


def _model_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": bundle["model_name"],
        "last_issue": bundle["front"]["last_issue"],
        "feature_groups": bundle.get("feature_groups", list(FEATURE_GROUPS)),
        "model_preset": bundle.get("model_preset", "standard"),
        "feature_count": len(bundle["front"]["feature_columns"]),
        "front_metrics": bundle["front"]["metrics"],
        "back_metrics": bundle["back"]["metrics"],
    }


def _lab_custom_model_path(params: dict[str, Any]) -> Path:
    signature = {
        "model": params["model"],
        "window": params["train_window"],
        "features": sorted(params["feature_groups"]),
        "preset": params["model_preset"],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    window = params["train_window"] or "all"
    return MODEL_DIR / "lab" / f"{params['model']}_{window}_{digest}.joblib"


def _prediction_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["tickets"] = json.loads(data.pop("tickets_json"))
    return data


def _random_ticket_dicts(count: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng()
    tickets = []
    for _ in range(count):
        front = tuple(sorted(rng.choice(FRONT_NUMBERS, size=FRONT_PICK, replace=False)))
        back = tuple(sorted(rng.choice(BACK_NUMBERS, size=BACK_PICK, replace=False)))
        tickets.append({"front": _format_numbers(front), "back": _format_numbers(back), "score": None})
    return tickets


def _parse_numbers(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split())


def _favorite_numbers(value: Any) -> tuple[int, ...]:
    if isinstance(value, list):
        return tuple(int(part) for part in value)
    return _parse_numbers(str(value))


def _evaluate_favorite_groups(history: pd.DataFrame, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues = history["issue"].map(_issue_key).to_numpy()
    output = []
    for group in groups:
        group_id = str(group.get("id", ""))
        after_issue = group.get("generatedAfterIssue")
        after_key = _issue_key(after_issue) if after_issue else -1
        if after_key < 0:
            output.append({"id": group_id, "status": "untracked"})
            continue
        target_idx = int(np.searchsorted(issues, after_key, side="right"))
        if target_idx >= len(history):
            output.append({"id": group_id, "status": "pending"})
            continue

        target = history.iloc[target_idx]
        target_front = {int(target[f"front{i}"]) for i in range(1, 6)}
        target_back = {int(target[f"back{i}"]) for i in range(1, 3)}
        ticket_results = []
        for ticket in (group.get("tickets") or [])[:20]:
            front = _favorite_numbers(ticket.get("front", ticket.get("front_text", "")))
            back = _favorite_numbers(ticket.get("back", ticket.get("back_text", "")))
            if len(front) != FRONT_PICK or len(back) != BACK_PICK:
                continue
            hit = evaluate_ticket(front, back, target)
            ticket_results.append(
                {
                    "front_hits": hit.front_hits,
                    "back_hits": hit.back_hits,
                    "front_matches": sorted(set(front) & target_front),
                    "back_matches": sorted(set(back) & target_back),
                    "prize_level": hit.prize_level,
                    "prize_label": _prize_label(hit.prize_level),
                }
            )
        best = min(
            (item["prize_level"] for item in ticket_results if item["prize_level"] is not None),
            default=None,
        )
        output.append(
            {
                "id": group_id,
                "status": "evaluated",
                "target": _draw_to_dict(target),
                "tickets": ticket_results,
                "best_prize": best,
                "best_label": _prize_label(best),
            }
        )
    return output


def _prize_label(level: int | None) -> str:
    labels = {
        1: "一等奖",
        2: "二等奖",
        3: "三等奖",
        4: "四等奖",
        5: "五等奖",
        6: "六等奖",
        7: "七等奖",
        8: "八等奖",
        9: "九等奖",
    }
    return labels.get(level, "未中奖")


def _format_numbers(numbers: tuple[int, ...]) -> str:
    return " ".join(f"{n:02d}" for n in numbers)


def _model_path(model_name: str, train_window: int | None = None) -> Path:
    suffix = "all" if train_window is None else f"w{int(train_window)}"
    if suffix == "all":
        return MODEL_DIR / f"{model_name}.joblib"
    return MODEL_DIR / f"{model_name}_{suffix}.joblib"


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, "", "all", "none", 0, "0"):
        return None
    return int(value)


def _resolve_train_window(history: pd.DataFrame, payload: dict[str, Any]) -> int | None:
    months = _parse_optional_int(payload.get("trainMonths"))
    if months:
        dates = pd.to_datetime(history["date"], errors="coerce")
        if dates.notna().any():
            cutoff = dates.max() - pd.DateOffset(months=months)
            count = int((dates >= cutoff).sum())
            return max(count, 40)
    return _parse_optional_int(payload.get("trainWindow"))


def _parse_window_value(value: Any) -> int | None:
    if isinstance(value, str) and value.strip().lower() in {"all", "none", "full", "0"}:
        return None
    if value in (None, "", 0):
        return None
    return int(value)


def _issue_key(issue: object) -> int:
    digits = "".join(ch for ch in str(issue) if ch.isdigit())
    return int(digits) if digits else -1


def _scalar(sql: str) -> int:
    with connect() as db:
        return int(db.execute(sql).fetchone()[0])


def _log_refresh(result: dict[str, Any]) -> None:
    with connect() as db:
        db.execute(
            """
            insert into refresh_log
            (created_at, ok, message, before_count, after_count, latest_issue)
            values (?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                int(bool(result["ok"])),
                result["message"],
                result["before_count"],
                result["after_count"],
                result["latest_issue"],
            ),
        )


def _lab_models() -> list[dict[str, Any]]:
    models = []
    for model_id, (label, description, speed) in LAB_MODEL_META.items():
        available = True
        reason = None
        if model_id in {"xgboost", "lightgbm"} and importlib.util.find_spec(model_id) is None:
            available = False
            reason = "服务器未安装可选依赖"
        models.append(
            {
                "id": model_id,
                "label": label,
                "description": description,
                "speed": speed,
                "available": available,
                "reason": reason,
            }
        )
    models.append(
        {
            "id": "lstm",
            "label": "LSTM",
            "description": "原项目仅有占位脚本，尚无真实可验证实现",
            "speed": "slow",
            "available": False,
            "reason": "待完成序列模型和模型能力评估后开放",
        }
    )
    return models


def _validate_lab_model(value: Any) -> str:
    model = str(value)
    if model not in MODEL_NAMES:
        raise ValueError(f"不支持的模型: {model}")
    if model in {"xgboost", "lightgbm"} and importlib.util.find_spec(model) is None:
        raise ValueError(f"{model} 尚未安装")
    return model


def _lab_window(value: Any) -> int | None:
    if value in (None, "", 0, "0", "all"):
        return None
    window = int(value)
    if window < 50:
        raise ValueError("训练窗口不能少于50期")
    return min(window, 5000)


def _lab_source_mode(value: Any) -> str:
    source = str(value or "model")
    if source not in {"model", "quick_model", "custom_model"}:
        raise ValueError("不支持的生成来源")
    return source


def _lab_custom_params(payload: dict[str, Any]) -> dict[str, Any]:
    raw_groups = payload.get("featureGroups")
    groups = list(FEATURE_GROUPS) if raw_groups is None else list(dict.fromkeys(str(item) for item in raw_groups))
    if not groups:
        raise ValueError("至少选择一组学习特征")
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"不支持的特征组: {', '.join(unknown)}")

    preset = str(payload.get("modelPreset", "standard"))
    if preset not in {"conservative", "standard", "exploratory"}:
        raise ValueError("不支持的模型复杂度")
    objective = str(payload.get("objective", "balanced"))
    if objective not in {"balanced", "score", "structure", "coverage"}:
        raise ValueError("不支持的组合目标")
    return {
        "feature_groups": groups,
        "model_preset": preset,
        "objective": objective,
        "structure_weight": _clamp_optional_float(payload.get("structureWeight"), 0.0, 0.8),
        "temperature": _clamp_optional_float(payload.get("temperature"), 0.5, 1.5),
        "diversity_weight": _clamp_optional_float(payload.get("diversityWeight"), 0.0, 1.0),
    }


def _clamp_optional_float(value: Any, low: float, high: float) -> float | None:
    if value in (None, ""):
        return None
    return min(max(float(value), low), high)


def _lab_default_recommendation(cached: dict[str, Any] | None, latest_issue: str) -> dict[str, Any]:
    if cached and cached.get("latest_issue") == latest_issue:
        best = next((row for row in cached.get("rows", []) if not row.get("error")), None)
        if best:
            return {
                "model": best["model"],
                "model_label": LAB_MODEL_META.get(best["model"], (best["model"], "", ""))[0],
                "train_window": best.get("train_window_draws"),
                "score": best.get("score"),
                "source": "strict_holdout",
                "stale": False,
                "test_draws": cached.get("params", {}).get("test_draws"),
            }
    return {
        "model": "logistic",
        "model_label": LAB_MODEL_META["logistic"][0],
        "train_window": 300,
        "score": None,
        "source": "fast_baseline",
        "stale": True,
        "test_draws": None,
    }


def _create_lab_job(job_type: str, params: dict[str, Any], message: str, owner_key: str) -> str:
    job_id = uuid.uuid4().hex
    now = _now()
    with connect() as db:
        db.execute(
            """
            insert into lab_jobs
            (id, job_type, status, progress, message, params_json, created_at, updated_at, owner_key)
            values (?, ?, 'queued', 0, ?, ?, ?, ?, ?)
            """,
            (job_id, job_type, message, json.dumps(params, ensure_ascii=False), now, now, owner_key),
        )
    return job_id


def _start_lab_job(job_id: str, runner, params: dict[str, Any]) -> bool:
    if not _JOB_SLOTS.acquire(blocking=False):
        _update_lab_job(
            job_id,
            status="failed",
            progress=100,
            message="任务队列已满",
            error="job queue capacity exceeded",
        )
        return False

    def target() -> None:
        try:
            runner(job_id, params)
        except Exception as exc:
            LOGGER.exception("lab_job_failed job_id=%s job_type=%s", job_id, params.get("model"))
            _update_lab_job(job_id, status="failed", progress=100, message="任务失败", error=str(exc))
        finally:
            _JOB_SLOTS.release()

    _JOB_EXECUTOR.submit(target)
    return True


def _update_lab_job(
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result: Any = None,
    error: str | None = None,
) -> None:
    fields = ["updated_at = ?"]
    values: list[Any] = [_now()]
    for column, value in (("status", status), ("progress", progress), ("message", message), ("error", error)):
        if value is not None:
            fields.append(f"{column} = ?")
            values.append(value)
    if result is not None:
        fields.append("result_json = ?")
        values.append(json.dumps(_json_clean(result), ensure_ascii=False))
    values.append(job_id)
    with connect() as db:
        db.execute(f"update lab_jobs set {', '.join(fields)} where id = ?", values)


def _lab_job(job_id: str, owner_key: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            "select * from lab_jobs where id = ? and owner_key = ?",
            (job_id, owner_key),
        ).fetchone()
    return _lab_job_row(row) if row else None


def _lab_jobs(limit: int = 10, owner_key: str = "local-development") -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "select * from lab_jobs where owner_key = ? order by created_at desc limit ?",
            (owner_key, limit),
        ).fetchall()
    return [_lab_job_row(row) for row in rows]


def _lab_generation_history(limit: int = 30, owner_key: str = "local-development") -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select id, created_at, params_json, result_json
            from lab_jobs
            where owner_key = ? and job_type = 'generate'
              and status = 'completed' and result_json is not null
            order by created_at desc
            limit ?
            """,
            (owner_key, limit * 4),
        ).fetchall()
    output = []
    for row in rows:
        params = json.loads(row["params_json"])
        if not params.get("save", True):
            continue
        result = json.loads(row["result_json"])
        output.append(
            {
                "id": row["id"],
                "created_at": row["created_at"],
                "source_mode": params.get("source_mode", "model"),
                "model": result.get("model", params.get("model")),
                "model_label": result.get("model_label", params.get("model")),
                "train_window": result.get("train_window", params.get("train_window")),
                "feature_groups": result.get("feature_groups", params.get("feature_groups", [])),
                "model_preset": result.get("model_preset", params.get("model_preset", "standard")),
                "objective": result.get("objective", params.get("objective", "balanced")),
                "trained_until_issue": result.get("trained_until_issue"),
                "training_mode": result.get("training_mode", "unknown"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "tickets": result.get("tickets", []),
            }
        )
        if len(output) >= limit:
            break
    return output


def _lab_job_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["params"] = json.loads(data.pop("params_json"))
    data["params"].pop("_owner_key", None)
    data.pop("owner_key", None)
    data["result"] = json.loads(data.pop("result_json")) if data.get("result_json") else None
    data.pop("result_json", None)
    return data


def _run_lab_generate(job_id: str, params: dict[str, Any]) -> None:
    with _LAB_COMPUTE_LOCK:
        started_at = time.perf_counter()
        _update_lab_job(job_id, status="running", progress=10, message="准备历史特征")
        history = _safe_history(required=True)
        _update_lab_job(job_id, progress=25, message="训练或读取模型")
        model_path = _lab_custom_model_path(params)
        bundle = None
        cache_hit = False
        if model_path.exists():
            cached_bundle = load_model(model_path)
            if str(cached_bundle["front"].get("last_issue")) == str(history.iloc[-1]["issue"]):
                bundle = cached_bundle
                cache_hit = True
        if bundle is None:
            bundle = train_model_bundle(
                history,
                model_name=params["model"],
                train_window_draws=params["train_window"],
                feature_groups=params["feature_groups"],
                model_preset=params["model_preset"],
            )
            save_model(bundle, model_path)
        _update_lab_job(job_id, progress=65, message="计算号码相对评分")
        probabilities = predict_probabilities(bundle, history)
        _update_lab_job(job_id, progress=78, message="筛选组合候选池")
        context = history if params["train_window"] is None else history.tail(params["train_window"])
        tickets = [
            ticket.as_dict()
            for ticket in generate_tickets(
                probabilities,
                context,
                candidates=params["candidates"],
                top_k=params["tickets"],
                random_state=int(str(history.iloc[-1]["issue"]) or 42),
                objective=params["objective"],
                structure_weight=params["structure_weight"],
                temperature=params["temperature"],
                diversity_weight=params["diversity_weight"],
            )
        ]
        probability_view = {}
        for zone in ("front", "back"):
            rows = probabilities[probabilities["zone"] == zone].sort_values("probability", ascending=False)
            maximum = max(float(rows["probability"].max()), 1e-9)
            probability_view[zone] = [
                {
                    "number": int(row["number"]),
                    "number_text": f"{int(row['number']):02d}",
                    "probability": round(float(row["probability"]), 4),
                    "relative_score": round(float(row["probability"]) / maximum * 100, 1),
                    "rank": rank,
                }
                for rank, (_, row) in enumerate(rows.iterrows(), start=1)
            ]

        prediction_id = None
        if params["save"]:
            prediction_id = save_prediction(
                mode=params.get("source_mode", "model"),
                model_name=params["model"],
                trained_until_issue=str(history.iloc[-1]["issue"]),
                tickets=tickets,
                owner_key=params["_owner_key"],
            )
        result = {
            "prediction_id": prediction_id,
            "model": params["model"],
            "model_label": LAB_MODEL_META[params["model"]][0],
            "train_window": params["train_window"],
            "feature_groups": params["feature_groups"],
            "model_preset": params["model_preset"],
            "objective": params["objective"],
            "training_mode": "cached" if cache_hit else "trained",
            "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            "generation_settings": {
                "structure_weight": params["structure_weight"],
                "temperature": params["temperature"],
                "diversity_weight": params["diversity_weight"],
                "candidates": params["candidates"],
            },
            "trained_until_issue": str(history.iloc[-1]["issue"]),
            "tickets": tickets,
            "scores": probability_view,
            "diagnostics": _model_summary(bundle),
        }
        _update_lab_job(job_id, status="completed", progress=100, message="实验组合已生成", result=result)


def _run_lab_backtest(job_id: str, params: dict[str, Any]) -> None:
    with _LAB_COMPUTE_LOCK:
        _update_lab_job(job_id, status="running", progress=10, message="划分训练集和测试集")
        history = _safe_history(required=True)
        _update_lab_job(job_id, progress=25, message="训练能力评估模型")
        results = holdout_evaluate(
            history,
            model_name=params["model"],
            train_until_issue=params["cutoff_issue"],
            tickets_per_draw=params["tickets"],
            candidates=params["candidates"],
            max_test_draws=params["test_draws"],
            train_window_draws=params["train_window"],
            feature_groups=params["feature_groups"],
            model_preset=params["model_preset"],
            objective=params["objective"],
            structure_weight=params["structure_weight"],
            temperature=params["temperature"],
            diversity_weight=params["diversity_weight"],
        )
        _update_lab_job(job_id, progress=90, message="与随机基线比较")
        summary = summarize_results(results).replace({np.nan: None}).to_dict(orient="records")
        result = {
            "model": params["model"],
            "model_label": LAB_MODEL_META[params["model"]][0],
            "train_window": params["train_window"],
            "feature_groups": params["feature_groups"],
            "model_preset": params["model_preset"],
            "objective": params["objective"],
            "cutoff_issue": params["cutoff_issue"],
            "test_draws": int(results["issue"].nunique()),
            "summary": summary,
        }
        _update_lab_job(job_id, status="completed", progress=100, message="模型能力评估完成", result=result)


def _run_lab_recommend(job_id: str, params: dict[str, Any]) -> None:
    with _LAB_COMPUTE_LOCK:
        history = _safe_history(required=True)
        cutoff_index = max(40, len(history) - params["test_draws"] - 1)
        cutoff = str(history.iloc[cutoff_index]["issue"])
        _update_lab_job(job_id, status="running", progress=10, message="比较模型与训练窗口")
        recommendations = recommend_models(
            history,
            model_names=params["models"],
            train_windows=params["windows"],
            train_until_issue=cutoff,
            tickets_per_draw=params["tickets"],
            candidates=params["candidates"],
            max_test_draws=params["test_draws"],
        )
        rows = recommendations.replace({np.nan: None}).to_dict(orient="records")
        cache = _json_clean(
            {
                "latest_issue": str(history.iloc[-1]["issue"]),
                "created_at": _now(),
                "params": params,
                "rows": rows,
            }
        )
        _save_recommendation_cache(cache)
        _update_lab_job(
            job_id,
            status="completed",
            progress=100,
            message="推荐实验完成",
            result={"cutoff_issue": cutoff, "rows": rows},
        )


def _lab_predictions(limit: int = 8, owner_key: str = "local-development") -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select id, created_at, mode, model_name, trained_until_issue, target_issue,
                   tickets_json, status, evaluated_at, best_front_hits, best_back_hits, best_prize
            from predictions
            where owner_key = ?
            order by id desc
            limit ?
            """,
            (owner_key, limit),
        ).fetchall()
    return [_prediction_row(row) for row in rows]


def _lab_metrics(owner_key: str = "local-development") -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            """
            select p.mode, p.model_name, r.front_hits, r.back_hits, r.prize_level
            from prediction_results r
            join predictions p on p.id = r.prediction_id
            where p.owner_key = ?
            """
            ,
            (owner_key,),
        ).fetchall()
    if not rows:
        return []
    frame = pd.DataFrame([dict(row) for row in rows])
    output = []
    for (mode, model), group in frame.groupby(["mode", "model_name"]):
        output.append(
            {
                "mode": mode,
                "model": model,
                "tickets": int(len(group)),
                "avg_front_hits": round(float(group["front_hits"].mean()), 3),
                "avg_back_hits": round(float(group["back_hits"].mean()), 3),
                "prize_rate": round(float(group["prize_level"].notna().mean()), 4),
            }
        )
    return output


def _last_refresh() -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            """
            select created_at, ok, message, before_count, after_count, latest_issue
            from refresh_log
            order by id desc
            limit 1
            """
        ).fetchone()
    return dict(row) if row else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_recommend_cutoff(history: pd.DataFrame) -> str:
    if len(history) <= 6:
        return str(history.iloc[max(0, len(history) - 2)]["issue"])
    return str(history.iloc[-4]["issue"])


def _load_recommendation_cache() -> dict[str, Any] | None:
    if not RECOMMENDATION_CACHE_PATH.exists():
        return None
    try:
        return json.loads(RECOMMENDATION_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_recommendation_cache(data: dict[str, Any]) -> None:
    RECOMMENDATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECOMMENDATION_CACHE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    return value


if __name__ == "__main__":
    create_app().run(debug=True)

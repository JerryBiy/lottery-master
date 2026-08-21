from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_prediction_frame, build_training_frame

FEATURE_GROUPS = {
    "trend": (
        "count_",
        "rate_",
        "ema_",
        "frequency",
        "last5_trend",
        "appeared_last",
        "appeared_previous",
    ),
    "heat": ("heat_", "cold_", "is_hot_", "is_cold_", "omission"),
    "cycle": ("avg_gap", "gap_vs_avg", "consecutive_streak", "draw_count_seen", "total_count"),
    "properties": ("number", "number_norm", "is_odd", "is_prime", "is_small", "tail", "tail_norm", "third"),
    "context": ("last_sum", "last_span", "last_odd_count", "last_low_count"),
    "relations": ("mirror", "mirror_last", "neighbor_last"),
}

MODEL_NAMES = (
    "logistic",
    "hist_gradient_boosting",
    "gradient_boosting",
    "random_forest",
    "extra_trees",
    "mlp",
    "xgboost",
    "lightgbm",
)


def make_estimator(model_name: str, random_state: int = 42, preset: str = "standard") -> Any:
    if preset not in {"conservative", "standard", "exploratory"}:
        raise ValueError(f"unknown model preset: {preset}")
    complexity = {"conservative": 0, "standard": 1, "exploratory": 2}[preset]
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=(220, 400, 600)[complexity],
            min_samples_leaf=(10, 6, 3)[complexity],
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=(250, 500, 700)[complexity],
            min_samples_leaf=(8, 4, 2)[complexity],
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=(0.25, 1.0, 4.0)[complexity],
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=(140, 220, 320)[complexity],
            learning_rate=(0.035, 0.04, 0.045)[complexity],
            l2_regularization=(0.2, 0.05, 0.01)[complexity],
            max_leaf_nodes=(9, 15, 25)[complexity],
            random_state=random_state,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=(120, 180, 260)[complexity],
            learning_rate=0.035,
            max_depth=(1, 2, 3)[complexity],
            min_samples_leaf=(18, 12, 6)[complexity],
            random_state=random_state,
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(48, 24),
                        alpha=(0.05, 0.01, 0.002)[complexity],
                        learning_rate_init=0.001,
                        max_iter=400,
                        early_stopping=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
        )
    raise ValueError(f"unknown model: {model_name}")


def train_zone_model(
    history: pd.DataFrame,
    zone: str,
    model_name: str = "random_forest",
    min_history: int = 30,
    feature_groups: list[str] | None = None,
    model_preset: str = "standard",
) -> dict[str, Any]:
    frame = build_training_frame(history, zone=zone, min_history=min_history)
    feature_columns = select_feature_columns(frame.X.columns, feature_groups)
    estimator = make_estimator(model_name, preset=model_preset)
    estimator.fit(frame.X[feature_columns], frame.y)
    metrics = in_sample_metrics(estimator, frame.X[feature_columns], frame.y)
    return {
        "zone": zone,
        "model_name": model_name,
        "estimator": estimator,
        "feature_columns": feature_columns,
        "feature_groups": feature_groups or list(FEATURE_GROUPS),
        "model_preset": model_preset,
        "metrics": metrics,
        "train_draws": len(history),
        "last_issue": str(history.iloc[-1]["issue"]),
    }


def train_model_bundle(
    history: pd.DataFrame,
    model_name: str = "random_forest",
    min_history: int = 30,
    train_window_draws: int | None = None,
    feature_groups: list[str] | None = None,
    model_preset: str = "standard",
) -> dict[str, Any]:
    train_history = _window_history(history, train_window_draws)
    return {
        "model_name": model_name,
        "train_window_draws": train_window_draws,
        "feature_groups": feature_groups or list(FEATURE_GROUPS),
        "model_preset": model_preset,
        "front": train_zone_model(train_history, "front", model_name, min_history, feature_groups, model_preset),
        "back": train_zone_model(train_history, "back", model_name, min_history, feature_groups, model_preset),
    }


def predict_probabilities(bundle: dict[str, Any], history: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    context_history = _window_history(history, bundle.get("train_window_draws"))
    for zone in ("front", "back"):
        X, meta = build_prediction_frame(context_history, zone)
        columns = bundle[zone]["feature_columns"]
        proba = _positive_probability(bundle[zone]["estimator"], X[columns])
        item = meta.copy()
        item["probability"] = proba
        outputs.append(item)
    return pd.concat(outputs, ignore_index=True)


def _window_history(history: pd.DataFrame, train_window_draws: int | None) -> pd.DataFrame:
    if train_window_draws is None or int(train_window_draws) <= 0 or int(train_window_draws) >= len(history):
        return history.reset_index(drop=True)
    return history.tail(int(train_window_draws)).reset_index(drop=True)


def select_feature_columns(columns, feature_groups: list[str] | None = None) -> list[str]:
    groups = list(FEATURE_GROUPS) if feature_groups is None else list(dict.fromkeys(feature_groups))
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups: {unknown}")
    selected = []
    for column in columns:
        if any(
            column == pattern or column.startswith(pattern)
            for group in groups
            for pattern in FEATURE_GROUPS[group]
        ):
            selected.append(column)
    if not selected:
        raise ValueError("feature selection produced no columns")
    return selected


def save_model(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def in_sample_metrics(estimator: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = _positive_probability(estimator, X)
    metrics = {"brier": float(brier_score_loss(y, proba))}
    if len(set(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y, proba))
        metrics["avg_precision"] = float(average_precision_score(y, proba))
    return metrics


def _positive_probability(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    return 1 / (1 + np.exp(-scores))

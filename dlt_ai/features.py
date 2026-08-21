from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BACK_NUMBERS, EMA_SPANS, FRONT_NUMBERS, ROLLING_WINDOWS


@dataclass(frozen=True)
class FeatureFrame:
    X: pd.DataFrame
    y: pd.Series | None
    meta: pd.DataFrame


PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}


def build_training_frame(history: pd.DataFrame, zone: str, min_history: int = 30) -> FeatureFrame:
    if len(history) <= min_history:
        raise ValueError(f"need more than {min_history} draws to build features")

    presence = _presence_matrix(history, zone)
    numbers = _numbers(zone)
    rows: list[pd.DataFrame] = []
    labels: list[int] = []
    meta_rows: list[pd.DataFrame] = []

    state = _empty_state(zone)
    prefix = _prefix_counts(presence)
    for draw_idx in range(len(history)):
        if draw_idx >= min_history:
            X = _features_from_state(state, prefix, draw_idx, zone)
            meta = pd.DataFrame({"zone": zone, "number": list(numbers)})
            meta["target_issue"] = str(history.iloc[draw_idx]["issue"])
            meta["target_date"] = history.iloc[draw_idx]["date"]
            rows.append(X)
            labels.extend(presence[draw_idx].astype(int).tolist())
            meta_rows.append(meta)
        _update_state(state, presence[draw_idx], draw_idx)

    return FeatureFrame(
        X=pd.concat(rows, ignore_index=True),
        y=pd.Series(labels, name="appeared"),
        meta=pd.concat(meta_rows, ignore_index=True),
    )


def build_prediction_frame(history: pd.DataFrame, zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    presence = _presence_matrix(history, zone)
    prefix = _prefix_counts(presence)
    state = _empty_state(zone)
    for draw_idx in range(len(history)):
        _update_state(state, presence[draw_idx], draw_idx)
    X = _features_from_state(state, prefix, len(history), zone)
    meta = pd.DataFrame({"zone": zone, "number": list(_numbers(zone))})
    return X, meta


def _features_from_state(state: dict, prefix: np.ndarray, draw_count: int, zone: str) -> pd.DataFrame:
    numbers = np.array(_numbers(zone), dtype=int)
    max_number = len(numbers)
    idx = np.arange(max_number)
    last_draw = state["last_draw"]
    previous_draw = state["previous_draw"]
    counts = state["counts"].astype(float)
    omission = np.where(state["last_seen"] >= 0, draw_count - state["last_seen"] - 1, draw_count)
    avg_gap = np.where(state["gap_count"] > 0, state["gap_sum"] / np.maximum(1, state["gap_count"]), float(draw_count))

    rows: dict[str, np.ndarray] = {
        "number": numbers,
        "number_norm": numbers / max_number,
        "tail": numbers % 10,
        "tail_norm": (numbers % 10) / 9,
        "is_odd": numbers % 2,
        "is_prime": np.array([int(n in PRIMES) for n in numbers]),
        "is_small": (numbers <= max_number / 2).astype(int),
        "third": np.minimum(2, ((numbers - 1) / (max_number / 3)).astype(int)),
        "mirror": max_number + 1 - numbers,
        "appeared_last": last_draw.astype(int),
        "appeared_previous": previous_draw.astype(int),
        "neighbor_last": _neighbor_indicator(last_draw),
        "mirror_last": last_draw[::-1].astype(int),
        "draw_count_seen": np.full(max_number, draw_count),
        "total_count": counts.astype(int),
        "frequency": counts / max(1, draw_count),
        "omission": omission.astype(int),
        "consecutive_streak": state["streak"].astype(int),
        "avg_gap": avg_gap.astype(float),
        "gap_vs_avg": omission - avg_gap,
    }

    for window in ROLLING_WINDOWS:
        start = max(0, draw_count - window)
        recent_count = prefix[draw_count] - prefix[start]
        denom = max(1, draw_count - start)
        recent_rate = recent_count / denom
        expected_rate = _expected_rate(zone)
        zscore = (recent_rate - expected_rate) / np.sqrt(max(expected_rate * (1 - expected_rate) / denom, 1e-9))
        rows[f"count_{window}"] = recent_count.astype(int)
        rows[f"rate_{window}"] = recent_rate
        rows[f"heat_zscore_{window}"] = zscore
        rows[f"heat_rank_{window}"] = _descending_rank(recent_rate)
        rows[f"cold_rank_{window}"] = _ascending_rank(recent_rate)
        rows[f"is_hot_{window}"] = (recent_rate >= np.quantile(recent_rate, 0.75)).astype(int)
        rows[f"is_cold_{window}"] = (recent_rate <= np.quantile(recent_rate, 0.25)).astype(int)

    if draw_count >= 20:
        last5 = prefix[draw_count] - prefix[draw_count - 5]
        last20 = prefix[draw_count] - prefix[draw_count - 20]
        rows["last5_trend"] = last5 / 5 - last20 / 20
    else:
        rows["last5_trend"] = np.zeros(max_number)

    for span in EMA_SPANS:
        rows[f"ema_{span}"] = state[f"ema_{span}"].astype(float)

    last_numbers = numbers[last_draw.astype(bool)]
    if len(last_numbers):
        last_sum = int(last_numbers.sum())
        last_span = int(last_numbers.max() - last_numbers.min())
        last_odd = int((last_numbers % 2).sum())
        last_low = int((last_numbers <= max_number / 2).sum())
    else:
        last_sum = 0
        last_span = 0
        last_odd = 0
        last_low = 0
    rows["last_sum"] = np.full(max_number, last_sum)
    rows["last_span"] = np.full(max_number, last_span)
    rows["last_odd_count"] = np.full(max_number, last_odd)
    rows["last_low_count"] = np.full(max_number, last_low)

    return pd.DataFrame(rows).sort_index(axis=1)


def _empty_state(zone: str) -> dict:
    size = len(_numbers(zone))
    state = {
        "counts": np.zeros(size, dtype=int),
        "last_seen": np.full(size, -1, dtype=int),
        "gap_sum": np.zeros(size, dtype=float),
        "gap_count": np.zeros(size, dtype=int),
        "streak": np.zeros(size, dtype=int),
        "last_draw": np.zeros(size, dtype=int),
        "previous_draw": np.zeros(size, dtype=int),
    }
    for span in EMA_SPANS:
        state[f"ema_{span}"] = np.zeros(size, dtype=float)
    return state


def _update_state(state: dict, appeared: np.ndarray, draw_idx: int) -> None:
    appeared = appeared.astype(int)
    previous_last_seen = state["last_seen"].copy()
    seen_mask = appeared == 1

    repeated = seen_mask & (previous_last_seen >= 0)
    state["gap_sum"][repeated] += draw_idx - previous_last_seen[repeated]
    state["gap_count"][repeated] += 1
    state["last_seen"][seen_mask] = draw_idx
    state["counts"] += appeared
    state["streak"] = np.where(seen_mask, state["streak"] + 1, 0)

    for span in EMA_SPANS:
        alpha = 2 / (span + 1)
        if draw_idx == 0:
            state[f"ema_{span}"] = appeared.astype(float)
        else:
            state[f"ema_{span}"] = alpha * appeared + (1 - alpha) * state[f"ema_{span}"]

    state["previous_draw"] = state["last_draw"].copy()
    state["last_draw"] = appeared.copy()


def _presence_matrix(history: pd.DataFrame, zone: str) -> np.ndarray:
    numbers = _numbers(zone)
    matrix = np.zeros((len(history), len(numbers)), dtype=int)
    columns = [f"front{i}" for i in range(1, 6)] if zone == "front" else [f"back{i}" for i in range(1, 3)]
    for row_idx, row in enumerate(history[columns].to_numpy(dtype=int)):
        matrix[row_idx, row - 1] = 1
    return matrix


def _prefix_counts(presence: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, presence.shape[1]), dtype=int), np.cumsum(presence, axis=0)])


def _neighbor_indicator(last_draw: np.ndarray) -> np.ndarray:
    left = np.r_[0, last_draw[:-1]]
    right = np.r_[last_draw[1:], 0]
    return ((left + right) > 0).astype(int)


def _expected_rate(zone: str) -> float:
    return 5 / 35 if zone == "front" else 2 / 12


def _descending_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _ascending_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _numbers(zone: str) -> tuple[int, ...]:
    if zone == "front":
        return FRONT_NUMBERS
    if zone == "back":
        return BACK_NUMBERS
    raise ValueError("zone must be 'front' or 'back'")


def _draw_numbers(row: pd.Series, zone: str) -> set[int]:
    if zone == "front":
        return {int(row[f"front{i}"]) for i in range(1, 6)}
    if zone == "back":
        return {int(row[f"back{i}"]) for i in range(1, 3)}
    raise ValueError("zone must be 'front' or 'back'")

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from .config import BACK_MAX, FRONT_MAX


WINDOW_OPTIONS = (30, 50, 100, 300)


@dataclass(frozen=True)
class PoolConfig:
    name: str
    prefix: str
    maximum: int
    picks: int


POOLS = {
    "front": PoolConfig("前区", "front", FRONT_MAX, 5),
    "back": PoolConfig("后区", "back", BACK_MAX, 2),
}


def draw_to_product_dict(row: pd.Series) -> dict[str, Any]:
    front = [int(row[f"front{i}"]) for i in range(1, 6)]
    back = [int(row[f"back{i}"]) for i in range(1, 3)]
    date = pd.to_datetime(row["date"])
    return {
        "issue": str(row["issue"]),
        "date": date.strftime("%Y-%m-%d"),
        "weekday": f"星期{'一二三四五六日'[date.weekday()]}",
        "front": front,
        "back": back,
        "front_text": " ".join(f"{number:02d}" for number in front),
        "back_text": " ".join(f"{number:02d}" for number in back),
        "front_sum": int(sum(front)),
        "front_span": int(max(front) - min(front)),
        "odd_count": int(sum(number % 2 for number in front)),
    }


def home_payload(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {
            "draw_count": 0,
            "latest": None,
            "recent": [],
            "quick_stats": {},
            "disclaimer": _disclaimer(),
        }

    recent = history.tail(6).iloc[::-1]
    latest = draw_to_product_dict(history.iloc[-1])
    window = history.tail(min(100, len(history)))
    front_values = window[[f"front{i}" for i in range(1, 6)]].to_numpy()
    back_values = window[[f"back{i}" for i in range(1, 3)]].to_numpy()
    front_counts = np.bincount(front_values.ravel(), minlength=FRONT_MAX + 1)[1:]
    back_counts = np.bincount(back_values.ravel(), minlength=BACK_MAX + 1)[1:]
    all_front_values = history[[f"front{i}" for i in range(1, 6)]].to_numpy()
    all_back_values = history[[f"back{i}" for i in range(1, 3)]].to_numpy()
    front_sum_summary = _comparison_summary(front_values.sum(axis=1), all_front_values.sum(axis=1))
    back_sum_summary = _comparison_summary(back_values.sum(axis=1), all_back_values.sum(axis=1))
    return {
        "draw_count": int(len(history)),
        "latest": latest,
        "recent": [draw_to_product_dict(row) for _, row in recent.iterrows()],
        "quick_stats": {
            "window": int(len(window)),
            "front_hot": _top_numbers(front_counts, 5),
            "back_hot": _top_numbers(back_counts, 3),
            "average_front_sum": front_sum_summary["average"],
            "average_back_sum": back_sum_summary["average"],
            "front_sum_level": front_sum_summary["level"],
            "front_sum_level_label": front_sum_summary["level_label"],
            "back_sum_level": back_sum_summary["level"],
            "back_sum_level_label": back_sum_summary["level_label"],
        },
        "disclaimer": _disclaimer(),
    }


def number_statistics(history: pd.DataFrame, window: int = 100) -> dict[str, Any]:
    window = max(10, min(int(window), len(history)))
    selected = history.tail(window)
    return {
        "window": window,
        "available_windows": [value for value in WINDOW_OPTIONS if value <= len(history)],
        "front": _pool_statistics(selected, POOLS["front"]),
        "back": _pool_statistics(selected, POOLS["back"]),
        "note": "冷热和遗漏仅描述历史，不改变下一期开奖概率。",
    }


def distribution_statistics(history: pd.DataFrame, window: int = 100) -> dict[str, Any]:
    window = max(10, min(int(window), len(history)))
    selected = history.tail(window)
    fronts = selected[[f"front{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    backs = selected[[f"back{i}" for i in range(1, 3)]].to_numpy(dtype=int)
    all_fronts = history[[f"front{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    all_backs = history[[f"back{i}" for i in range(1, 3)]].to_numpy(dtype=int)
    sums = fronts.sum(axis=1)
    back_sums = backs.sum(axis=1)
    all_sums = all_fronts.sum(axis=1)
    all_back_sums = all_backs.sum(axis=1)
    spans = fronts.max(axis=1) - fronts.min(axis=1)
    back_spans = backs.max(axis=1) - backs.min(axis=1)
    odds = (fronts % 2).sum(axis=1)
    large_counts = (fronts >= 18).sum(axis=1)
    prime_counts = np.isin(fronts, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]).sum(axis=1)
    back_odds = (backs % 2).sum(axis=1)
    back_large = (backs >= 7).sum(axis=1)
    zones = np.stack(
        [
            ((fronts >= 1) & (fronts <= 12)).sum(axis=1),
            ((fronts >= 13) & (fronts <= 24)).sum(axis=1),
            ((fronts >= 25) & (fronts <= 35)).sum(axis=1),
        ],
        axis=1,
    )
    return {
        "window": window,
        "front_sum": _comparison_summary(sums, all_sums),
        "back_sum": _comparison_summary(back_sums, all_back_sums),
        "front_span": _numeric_summary(spans),
        "back_span": _numeric_summary(back_spans),
        "front_sum_histogram": _histogram(
            sums,
            [(15, 44), (45, 59), (60, 74), (75, 89), (90, 104), (105, 119), (120, 134), (135, 150), (151, 165)],
        ),
        "back_sum_histogram": _histogram(back_sums, [(3, 6), (7, 10), (11, 14), (15, 18), (19, 23)]),
        "pattern_metrics": _pattern_metrics(history, window),
        "ratio_distributions": {
            "big_small": _ratio_distribution(large_counts, 5, "大", "小"),
            "prime_composite": _ratio_distribution(prime_counts, 5, "质", "合"),
            "back_odd_even": _ratio_distribution(back_odds, 2, "奇", "偶"),
            "back_big_small": _ratio_distribution(back_large, 2, "大", "小"),
            "route_012": _route_distribution(fronts),
        },
        "tail_frequency": {
            "front": _tail_frequency(fronts),
            "back": _tail_frequency(backs),
        },
        "position_statistics": _position_statistics(fronts),
        "gap_statistics": _gap_statistics(fronts),
        "top_pairs": _top_pairs(fronts, 10),
        "omission_matrix": {
            "front": _omission_matrix(history, POOLS["front"], 15),
            "back": _omission_matrix(history, POOLS["back"], 15),
        },
        "odd_distribution": [
            {"label": f"{odd}奇{5 - odd}偶", "count": int((odds == odd).sum())}
            for odd in range(6)
        ],
        "zone_distribution": [
            {
                "label": f"{a}:{b}:{c}",
                "count": int(((zones[:, 0] == a) & (zones[:, 1] == b) & (zones[:, 2] == c)).sum()),
            }
            for a, b, c in sorted(
                set(map(tuple, zones.tolist())),
                key=lambda item: int(((zones == item).all(axis=1)).sum()),
                reverse=True,
            )[:8]
        ],
        "five_zone_distribution": _five_zone_distribution(fronts),
        "recent_series": [
            {
                "issue": str(row["issue"]),
                "sum": int(sum(int(row[f"front{i}"]) for i in range(1, 6))),
                "back_sum": int(sum(int(row[f"back{i}"]) for i in range(1, 3))),
                "span": int(max(int(row[f"front{i}"]) for i in range(1, 6)) - min(int(row[f"front{i}"]) for i in range(1, 6))),
            }
            for _, row in selected.tail(20).iterrows()
        ],
    }


def generate_random_tickets(
    count: int,
    seed: int | None = None,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    count = max(1, min(int(count), 20))
    rng = np.random.default_rng(seed)
    if not constraints:
        tickets = []
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        while len(tickets) < count:
            front = tuple(sorted(rng.choice(np.arange(1, FRONT_MAX + 1), size=5, replace=False).tolist()))
            back = tuple(sorted(rng.choice(np.arange(1, BACK_MAX + 1), size=2, replace=False).tolist()))
            if (front, back) in seen:
                continue
            seen.add((front, back))
            tickets.append(
                {
                    "id": len(tickets) + 1,
                    "front": list(front),
                    "back": list(back),
                    "front_text": " ".join(f"{number:02d}" for number in front),
                    "back_text": " ".join(f"{number:02d}" for number in back),
                }
            )
        return tickets
    rules = _normalize_random_constraints(constraints or {})
    front_candidates = _combination_candidates(
        maximum=FRONT_MAX,
        picks=5,
        required=rules["front_required"],
        excluded=rules["front_excluded"],
        predicate=lambda values: _front_random_match(values, rules),
    )
    back_candidates = _combination_candidates(
        maximum=BACK_MAX,
        picks=2,
        required=rules["back_required"],
        excluded=rules["back_excluded"],
        predicate=lambda values: rules["back_odd_count"] is None
        or sum(value % 2 for value in values) == rules["back_odd_count"],
    )
    if not front_candidates:
        raise ValueError("当前前区条件没有可用组合，请减少必选、排除或结构限制")
    if not back_candidates:
        raise ValueError("当前后区条件没有可用组合，请减少必选、排除或奇偶限制")
    total_combinations = len(front_candidates) * len(back_candidates)
    if total_combinations < count:
        raise ValueError(f"当前条件只能生成 {total_combinations} 注不同号码，请减少注数或放宽条件")

    selected = rng.choice(total_combinations, size=count, replace=False)
    tickets = []
    for index, flat_index in enumerate(selected):
        front_index, back_index = divmod(int(flat_index), len(back_candidates))
        front = list(front_candidates[front_index])
        back = list(back_candidates[back_index])
        tickets.append(
            {
                "id": index + 1,
                "front": front,
                "back": back,
                "front_text": " ".join(f"{number:02d}" for number in front),
                "back_text": " ".join(f"{number:02d}" for number in back),
            }
        )
    return tickets


def random_candidate_counts(constraints: dict[str, Any] | None = None) -> dict[str, int]:
    if not constraints:
        front_count = comb(FRONT_MAX, 5)
        back_count = comb(BACK_MAX, 2)
        return {"front": front_count, "back": back_count, "tickets": front_count * back_count}
    rules = _normalize_random_constraints(constraints or {})
    fronts = _combination_candidates(
        FRONT_MAX,
        5,
        rules["front_required"],
        rules["front_excluded"],
        lambda values: _front_random_match(values, rules),
    )
    backs = _combination_candidates(
        BACK_MAX,
        2,
        rules["back_required"],
        rules["back_excluded"],
        lambda values: rules["back_odd_count"] is None
        or sum(value % 2 for value in values) == rules["back_odd_count"],
    )
    return {"front": len(fronts), "back": len(backs), "tickets": len(fronts) * len(backs)}


def _normalize_random_constraints(value: dict[str, Any]) -> dict[str, Any]:
    def numbers(key: str, maximum: int) -> tuple[int, ...]:
        raw = value.get(key) or []
        if not isinstance(raw, list):
            raise ValueError(f"{key} 必须是号码列表")
        parsed = tuple(sorted({int(number) for number in raw}))
        if any(number < 1 or number > maximum for number in parsed):
            raise ValueError(f"{key} 包含超出范围的号码")
        return parsed

    def optional_count(key: str, maximum: int) -> int | None:
        raw = value.get(key)
        if raw in (None, "", "any"):
            return None
        parsed = int(raw)
        if not 0 <= parsed <= maximum:
            raise ValueError(f"{key} 超出允许范围")
        return parsed

    front_required = numbers("frontRequired", FRONT_MAX)
    front_excluded = numbers("frontExcluded", FRONT_MAX)
    back_required = numbers("backRequired", BACK_MAX)
    back_excluded = numbers("backExcluded", BACK_MAX)
    _validate_required_excluded(front_required, front_excluded, 5, FRONT_MAX, "前区")
    _validate_required_excluded(back_required, back_excluded, 2, BACK_MAX, "后区")

    sum_min = int(value.get("frontSumMin", 15) or 15)
    sum_max = int(value.get("frontSumMax", 165) or 165)
    if not 15 <= sum_min <= sum_max <= 165:
        raise ValueError("前区和值范围必须在 15 至 165 之间")
    consecutive = str(value.get("consecutive", "any"))
    if consecutive not in {"any", "require", "avoid"}:
        raise ValueError("连号条件不正确")
    zone_mode = str(value.get("zoneMode", "any"))
    if zone_mode not in {"any", "cover", "max2"}:
        raise ValueError("三区条件不正确")
    return {
        "front_required": front_required,
        "front_excluded": front_excluded,
        "back_required": back_required,
        "back_excluded": back_excluded,
        "front_odd_count": optional_count("frontOddCount", 5),
        "front_big_count": optional_count("frontBigCount", 5),
        "back_odd_count": optional_count("backOddCount", 2),
        "front_sum_min": sum_min,
        "front_sum_max": sum_max,
        "consecutive": consecutive,
        "unique_tails": bool(value.get("uniqueTails", False)),
        "zone_mode": zone_mode,
    }


def _validate_required_excluded(
    required: tuple[int, ...],
    excluded: tuple[int, ...],
    picks: int,
    maximum: int,
    label: str,
) -> None:
    if set(required) & set(excluded):
        raise ValueError(f"{label}同一号码不能同时必选和排除")
    if len(required) > picks:
        raise ValueError(f"{label}最多必选 {picks} 个号码")
    if maximum - len(excluded) < picks:
        raise ValueError(f"{label}排除号码过多，剩余号码不足")


def _combination_candidates(
    maximum: int,
    picks: int,
    required: tuple[int, ...],
    excluded: tuple[int, ...],
    predicate,
) -> list[tuple[int, ...]]:
    required_set = set(required)
    available = [
        number
        for number in range(1, maximum + 1)
        if number not in required_set and number not in excluded
    ]
    needed = picks - len(required)
    return [
        values
        for extra in combinations(available, needed)
        if predicate(values := tuple(sorted((*required, *extra))))
    ]


def _front_random_match(values: tuple[int, ...], rules: dict[str, Any]) -> bool:
    odd_count = sum(value % 2 for value in values)
    big_count = sum(value >= 18 for value in values)
    total = sum(values)
    has_consecutive = any(right - left == 1 for left, right in zip(values, values[1:]))
    zone_counts = (
        sum(value <= 12 for value in values),
        sum(13 <= value <= 24 for value in values),
        sum(value >= 25 for value in values),
    )
    return (
        (rules["front_odd_count"] is None or odd_count == rules["front_odd_count"])
        and (rules["front_big_count"] is None or big_count == rules["front_big_count"])
        and rules["front_sum_min"] <= total <= rules["front_sum_max"]
        and (rules["consecutive"] == "any" or (rules["consecutive"] == "require") == has_consecutive)
        and (not rules["unique_tails"] or len({value % 10 for value in values}) == 5)
        and (rules["zone_mode"] != "cover" or all(count >= 1 for count in zone_counts))
        and (rules["zone_mode"] != "max2" or all(count <= 2 for count in zone_counts))
    )


def _pool_statistics(history: pd.DataFrame, config: PoolConfig) -> list[dict[str, Any]]:
    columns = [f"{config.prefix}{index}" for index in range(1, config.picks + 1)]
    values = history[columns].to_numpy(dtype=int)
    counts = np.bincount(values.ravel(), minlength=config.maximum + 1)[1:]
    expected = len(history) * config.picks / config.maximum
    standard_deviation = float(counts.std()) or 1.0
    rows = []
    for number in range(1, config.maximum + 1):
        appearances = np.where((values == number).any(axis=1))[0]
        omission = len(history) if not len(appearances) else len(history) - 1 - int(appearances[-1])
        gaps = np.diff(appearances) if len(appearances) > 1 else np.array([], dtype=int)
        current_streak = 0
        for row in values[::-1]:
            if number in row:
                current_streak += 1
            else:
                break
        observed_omissions = np.diff(np.concatenate(([-1], appearances, [len(history)]))) - 1
        z_score = (int(counts[number - 1]) - expected) / standard_deviation
        if z_score >= 0.75:
            temperature = "hot"
        elif z_score <= -0.75:
            temperature = "cold"
        else:
            temperature = "normal"
        rows.append(
            {
                "number": number,
                "number_text": f"{number:02d}",
                "count": int(counts[number - 1]),
                "rate": round(float(counts[number - 1] / len(history)), 4),
                "omission": int(omission),
                "temperature": temperature,
                "heat_score": round(float(z_score), 2),
                "average_gap": round(float(gaps.mean()), 1) if len(gaps) else None,
                "max_omission": int(observed_omissions.max()) if len(observed_omissions) else len(history),
                "current_streak": int(current_streak),
            }
        )
    return rows


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "average": round(float(values.mean()), 1),
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "median": round(float(np.median(values)), 1),
    }


def _comparison_summary(values: np.ndarray, benchmark_values: np.ndarray) -> dict[str, Any]:
    summary = _numeric_summary(values)
    benchmark = float(benchmark_values.mean())
    delta = float(values.mean() - benchmark)
    standard_error = float(benchmark_values.std() / np.sqrt(max(len(values), 1)))
    threshold = max(0.35, standard_error)
    if delta < -threshold:
        level, level_label = "low", "偏低"
    elif delta > threshold:
        level, level_label = "high", "偏高"
    else:
        level, level_label = "normal", "接近历史均值"
    return {
        **summary,
        "benchmark": round(benchmark, 1),
        "delta_from_history": round(delta, 1),
        "level": level,
        "level_label": level_label,
    }


def _histogram(values: np.ndarray, ranges: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {
            "label": f"{lower}-{upper}",
            "minimum": lower,
            "maximum": upper,
            "count": int(((values >= lower) & (values <= upper)).sum()),
            "rate": round(float(((values >= lower) & (values <= upper)).mean()), 4),
        }
        for lower, upper in ranges
    ]


def _ratio_distribution(counts: np.ndarray, picks: int, first: str, second: str) -> list[dict[str, Any]]:
    return [
        {
            "label": f"{count}{first}{picks - count}{second}",
            "count": int((counts == count).sum()),
        }
        for count in range(picks + 1)
    ]


def _route_distribution(fronts: np.ndarray) -> list[dict[str, Any]]:
    routes = fronts % 3
    structures = [
        (
            int((row == 0).sum()),
            int((row == 1).sum()),
            int((row == 2).sum()),
        )
        for row in routes
    ]
    counts: dict[tuple[int, int, int], int] = {}
    for structure in structures:
        counts[structure] = counts.get(structure, 0) + 1
    return [
        {"label": f"{key[0]}:{key[1]}:{key[2]}", "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


def _tail_frequency(values: np.ndarray) -> list[dict[str, Any]]:
    counts = np.bincount((values % 10).ravel(), minlength=10)
    maximum = max(int(counts.max()), 1)
    return [
        {
            "tail": tail,
            "label": f"{tail}尾",
            "count": int(counts[tail]),
            "percent": round(float(counts[tail] / maximum * 100), 1),
        }
        for tail in range(10)
    ]


def _position_statistics(fronts: np.ndarray) -> list[dict[str, Any]]:
    return [
        {
            "position": index + 1,
            "label": f"第{index + 1}位",
            "average": round(float(fronts[:, index].mean()), 1),
            "minimum": int(fronts[:, index].min()),
            "maximum": int(fronts[:, index].max()),
        }
        for index in range(fronts.shape[1])
    ]


def _gap_statistics(fronts: np.ndarray) -> list[dict[str, Any]]:
    gaps = np.diff(fronts, axis=1)
    return [
        {
            "position": index + 1,
            "label": f"间距{index + 1}",
            "average": round(float(gaps[:, index].mean()), 1),
            "median": round(float(np.median(gaps[:, index])), 1),
            "maximum": int(gaps[:, index].max()),
        }
        for index in range(gaps.shape[1])
    ]


def _top_pairs(fronts: np.ndarray, size: int) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], int] = {}
    for row in fronts:
        for pair in combinations(map(int, row), 2):
            counts[pair] = counts.get(pair, 0) + 1
    return [
        {
            "key": f"{pair[0]}-{pair[1]}",
            "first": pair[0],
            "second": pair[1],
            "first_text": f"{pair[0]:02d}",
            "second_text": f"{pair[1]:02d}",
            "count": count,
        }
        for pair, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:size]
    ]


def _five_zone_distribution(fronts: np.ndarray) -> list[dict[str, Any]]:
    zones = np.stack(
        [
            ((fronts >= lower) & (fronts <= upper)).sum(axis=1)
            for lower, upper in ((1, 7), (8, 14), (15, 21), (22, 28), (29, 35))
        ],
        axis=1,
    )
    structures: dict[tuple[int, ...], int] = {}
    for row in zones:
        key = tuple(map(int, row))
        structures[key] = structures.get(key, 0) + 1
    return [
        {"label": ":".join(map(str, key)), "count": value}
        for key, value in sorted(structures.items(), key=lambda item: (-item[1], item[0]))[:8]
    ]


def _omission_matrix(history: pd.DataFrame, config: PoolConfig, draws: int) -> list[dict[str, Any]]:
    columns = [f"{config.prefix}{index}" for index in range(1, config.picks + 1)]
    start = max(0, len(history) - draws)
    omissions = np.zeros(config.maximum + 1, dtype=int)
    for _, row in history.iloc[:start].iterrows():
        hits = {int(row[column]) for column in columns}
        for number in range(1, config.maximum + 1):
            omissions[number] = 0 if number in hits else omissions[number] + 1

    rows = []
    for _, row in history.iloc[start:].iterrows():
        hits = {int(row[column]) for column in columns}
        cells = []
        for number in range(1, config.maximum + 1):
            if number in hits:
                omissions[number] = 0
            else:
                omissions[number] += 1
            cells.append(
                {
                    "number": number,
                    "number_text": f"{number:02d}",
                    "hit": number in hits,
                    "omission": int(omissions[number]),
                }
            )
        rows.append({"issue": str(row["issue"]), "cells": cells})
    return rows


def _pattern_metrics(history: pd.DataFrame, window: int) -> list[dict[str, Any]]:
    all_fronts = history[[f"front{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    fronts = all_fronts[-window:]
    consecutive = np.any(np.diff(fronts, axis=1) == 1, axis=1)
    primes = np.isin(fronts, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31])
    large = fronts >= 18
    tail_diversity = np.array([len(set((row % 10).tolist())) for row in fronts])

    start = len(all_fronts) - window
    repeat_flags = []
    for index in range(max(1, start), len(all_fronts)):
        repeat_flags.append(bool(set(all_fronts[index]) & set(all_fronts[index - 1])))
    repeat_rate = float(np.mean(repeat_flags)) if repeat_flags else 0.0

    return [
        {
            "key": "consecutive_rate",
            "label": "含连号期数",
            "value": round(float(consecutive.mean()), 4),
            "display": f"{consecutive.mean() * 100:.1f}%",
        },
        {
            "key": "repeat_rate",
            "label": "含重号期数",
            "value": round(repeat_rate, 4),
            "display": f"{repeat_rate * 100:.1f}%",
        },
        {
            "key": "prime_average",
            "label": "平均质数个数",
            "value": round(float(primes.sum(axis=1).mean()), 2),
            "display": f"{primes.sum(axis=1).mean():.2f}",
        },
        {
            "key": "large_average",
            "label": "平均大号个数",
            "value": round(float(large.sum(axis=1).mean()), 2),
            "display": f"{large.sum(axis=1).mean():.2f}",
        },
        {
            "key": "tail_diversity",
            "label": "平均尾数种类",
            "value": round(float(tail_diversity.mean()), 2),
            "display": f"{tail_diversity.mean():.2f}",
        },
    ]


def _top_numbers(counts: np.ndarray, size: int) -> list[dict[str, int | str]]:
    indexes = np.argsort(counts)[::-1][:size]
    return [
        {"number": int(index + 1), "number_text": f"{index + 1:02d}", "count": int(counts[index])}
        for index in indexes
    ]


def _disclaimer() -> str:
    return "本工具仅展示历史数据与概率知识。每期开奖相互独立，历史统计不能提高中奖概率，请理性购彩。"

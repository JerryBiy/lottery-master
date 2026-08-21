from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BACK_NUMBERS, BACK_PICK, FRONT_NUMBERS, FRONT_PICK


@dataclass(frozen=True)
class Ticket:
    front: tuple[int, ...]
    back: tuple[int, ...]
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "front": " ".join(f"{n:02d}" for n in self.front),
            "back": " ".join(f"{n:02d}" for n in self.back),
            "score": self.score,
        }


def generate_tickets(
    probabilities: pd.DataFrame,
    history: pd.DataFrame,
    candidates: int = 100_000,
    top_k: int = 10,
    random_state: int = 42,
    objective: str = "balanced",
    structure_weight: float | None = None,
    temperature: float | None = None,
    diversity_weight: float | None = None,
) -> list[Ticket]:
    objective_defaults = {
        "balanced": (0.18, 1.0, 0.15),
        "score": (0.05, 0.78, 0.0),
        "structure": (0.38, 1.0, 0.10),
        "coverage": (0.15, 1.12, 0.75),
    }
    if objective not in objective_defaults:
        raise ValueError(f"unknown generation objective: {objective}")
    default_structure, default_temperature, default_diversity = objective_defaults[objective]
    structure_weight = default_structure if structure_weight is None else float(structure_weight)
    temperature = default_temperature if temperature is None else float(temperature)
    diversity_weight = default_diversity if diversity_weight is None else float(diversity_weight)

    rng = np.random.default_rng(random_state)
    front_probs = _prob_vector(probabilities, "front", FRONT_NUMBERS, temperature)
    back_probs = _prob_vector(probabilities, "back", BACK_NUMBERS, temperature)
    front_log = np.log(np.clip(_raw_prob_vector(probabilities, "front", FRONT_NUMBERS), 1e-9, None))
    back_log = np.log(np.clip(_raw_prob_vector(probabilities, "back", BACK_NUMBERS), 1e-9, None))
    profile = _history_profile(history)

    seen: dict[tuple[tuple[int, ...], tuple[int, ...]], Ticket] = {}
    for _ in range(candidates):
        front = tuple(sorted(rng.choice(FRONT_NUMBERS, size=FRONT_PICK, replace=False, p=front_probs)))
        back = tuple(sorted(rng.choice(BACK_NUMBERS, size=BACK_PICK, replace=False, p=back_probs)))
        key = (front, back)
        if key in seen:
            continue
        score = _ticket_score(front, back, front_log, back_log, profile, structure_weight)
        seen[key] = Ticket(front=front, back=back, score=score)

    ranked = sorted(seen.values(), key=lambda x: x.score, reverse=True)
    return _select_diverse_tickets(ranked, top_k, diversity_weight)


def _raw_prob_vector(probabilities: pd.DataFrame, zone: str, numbers: tuple[int, ...]) -> np.ndarray:
    zone_probs = probabilities[probabilities["zone"] == zone].set_index("number")["probability"]
    return np.array([float(zone_probs.loc[n]) for n in numbers], dtype=float)


def _prob_vector(
    probabilities: pd.DataFrame,
    zone: str,
    numbers: tuple[int, ...],
    temperature: float = 1.0,
) -> np.ndarray:
    values = np.clip(_raw_prob_vector(probabilities, zone, numbers), 1e-6, None)
    values = values ** (1.0 / max(0.2, temperature))
    return values / values.sum()


def _ticket_score(
    front: tuple[int, ...],
    back: tuple[int, ...],
    front_log: np.ndarray,
    back_log: np.ndarray,
    profile: dict[str, float],
    structure_weight: float = 0.18,
) -> float:
    log_prob = float(front_log[np.array(front) - 1].sum() + back_log[np.array(back) - 1].sum())

    front_sum = sum(front)
    span = max(front) - min(front)
    odd_count = sum(n % 2 for n in front)
    thirds = [
        sum(1 for n in front if 1 <= n <= 12),
        sum(1 for n in front if 13 <= n <= 24),
        sum(1 for n in front if 25 <= n <= 35),
    ]

    penalty = 0.0
    penalty += abs(front_sum - profile["front_sum_mean"]) / max(1.0, profile["front_sum_std"])
    penalty += abs(span - profile["front_span_mean"]) / max(1.0, profile["front_span_std"])
    penalty += abs(odd_count - profile["front_odd_mean"]) / max(1.0, profile["front_odd_std"])
    penalty += sum(abs(thirds[i] - profile[f"third_{i}_mean"]) for i in range(3)) * 0.35
    return float(log_prob - structure_weight * penalty)


def _select_diverse_tickets(ranked: list[Ticket], top_k: int, diversity_weight: float) -> list[Ticket]:
    if diversity_weight <= 0 or top_k <= 1:
        return ranked[:top_k]

    pool = ranked[: max(top_k * 60, 200)]
    selected: list[Ticket] = []
    remaining = list(pool)
    while remaining and len(selected) < top_k:
        if not selected:
            chosen = remaining[0]
        else:
            chosen = max(
                remaining,
                key=lambda ticket: ticket.score
                - diversity_weight * max(_ticket_overlap(ticket, other) for other in selected),
            )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _ticket_overlap(left: Ticket, right: Ticket) -> float:
    front_overlap = len(set(left.front) & set(right.front)) / FRONT_PICK
    back_overlap = len(set(left.back) & set(right.back)) / BACK_PICK
    return (front_overlap + back_overlap) / 2.0


def _history_profile(history: pd.DataFrame) -> dict[str, float]:
    front_cols = [f"front{i}" for i in range(1, 6)]
    fronts = history[front_cols].to_numpy(dtype=int)
    sums = fronts.sum(axis=1)
    spans = fronts.max(axis=1) - fronts.min(axis=1)
    odds = (fronts % 2).sum(axis=1)
    profile = {
        "front_sum_mean": float(sums.mean()),
        "front_sum_std": float(sums.std(ddof=0)),
        "front_span_mean": float(spans.mean()),
        "front_span_std": float(spans.std(ddof=0)),
        "front_odd_mean": float(odds.mean()),
        "front_odd_std": float(odds.std(ddof=0)),
    }
    for idx, bounds in enumerate(((1, 12), (13, 24), (25, 35))):
        low, high = bounds
        counts = ((fronts >= low) & (fronts <= high)).sum(axis=1)
        profile[f"third_{idx}_mean"] = float(counts.mean())
    return profile

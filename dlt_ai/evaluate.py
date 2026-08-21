from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BACK_NUMBERS, BACK_PICK, FRONT_NUMBERS, FRONT_PICK
from .models import predict_probabilities, train_model_bundle
from .optimize import generate_tickets


@dataclass(frozen=True)
class HitResult:
    front_hits: int
    back_hits: int
    prize_level: int | None


def evaluate_ticket(front: tuple[int, ...], back: tuple[int, ...], draw: pd.Series) -> HitResult:
    target_front = {int(draw[f"front{i}"]) for i in range(1, 6)}
    target_back = {int(draw[f"back{i}"]) for i in range(1, 3)}
    front_hits = len(set(front) & target_front)
    back_hits = len(set(back) & target_back)
    return HitResult(front_hits, back_hits, prize_level=prize_level(front_hits, back_hits))


def prize_level(front_hits: int, back_hits: int) -> int | None:
    if front_hits == 5 and back_hits == 2:
        return 1
    if front_hits == 5 and back_hits == 1:
        return 2
    if front_hits == 5 and back_hits == 0:
        return 3
    if front_hits == 4 and back_hits == 2:
        return 4
    if front_hits == 4 and back_hits == 1:
        return 5
    if front_hits == 3 and back_hits == 2:
        return 6
    if front_hits == 4 and back_hits == 0:
        return 7
    if (front_hits == 3 and back_hits == 1) or (front_hits == 2 and back_hits == 2):
        return 8
    if (front_hits == 3 and back_hits == 0) or (front_hits == 1 and back_hits == 2) or (front_hits == 2 and back_hits == 1) or (front_hits == 0 and back_hits == 2):
        return 9
    return None


def backtest(
    history: pd.DataFrame,
    model_name: str,
    train_until: str | None = None,
    train_until_issue: str | None = None,
    tickets_per_draw: int = 10,
    candidates: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    start_idx = _backtest_start_idx(history, train_until=train_until, train_until_issue=train_until_issue)
    if start_idx < 40:
        raise ValueError("backtest split leaves too little training history")

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    for draw_idx in range(start_idx, len(history)):
        train_history = history.iloc[:draw_idx].reset_index(drop=True)
        target = history.iloc[draw_idx]
        bundle = train_model_bundle(train_history, model_name=model_name)
        probabilities = predict_probabilities(bundle, train_history)
        tickets = generate_tickets(probabilities, train_history, candidates=candidates, top_k=tickets_per_draw, random_state=random_state + draw_idx)
        random_tickets = _random_tickets(rng, tickets_per_draw)

        rows.extend(_score_ticket_group(target, tickets, "model"))
        rows.extend(_score_ticket_group(target, random_tickets, "random"))
    return pd.DataFrame(rows)


def holdout_evaluate(
    history: pd.DataFrame,
    model_name: str,
    train_until_issue: str,
    tickets_per_draw: int = 10,
    candidates: int = 5_000,
    max_test_draws: int | None = None,
    train_window_draws: int | None = None,
    random_state: int = 42,
    feature_groups: list[str] | None = None,
    model_preset: str = "standard",
    objective: str = "balanced",
    structure_weight: float | None = None,
    temperature: float | None = None,
    diversity_weight: float | None = None,
) -> pd.DataFrame:
    """Train once before the cutoff, then evaluate each later draw vs random."""
    start_idx = _backtest_start_idx(history, train_until=None, train_until_issue=train_until_issue)
    if start_idx < 40:
        raise ValueError("holdout split leaves too little training history")
    end_idx = len(history) if max_test_draws is None else min(len(history), start_idx + max_test_draws)
    if end_idx <= start_idx:
        raise ValueError("holdout split leaves no test draws")

    train_history = _window_history(history.iloc[:start_idx], train_window_draws)
    bundle = train_model_bundle(
        train_history,
        model_name=model_name,
        train_window_draws=train_window_draws,
        feature_groups=feature_groups,
        model_preset=model_preset,
    )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for draw_idx in range(start_idx, end_idx):
        context_history = _window_history(history.iloc[:draw_idx], train_window_draws)
        target = history.iloc[draw_idx]
        probabilities = predict_probabilities(bundle, context_history)
        tickets = generate_tickets(
            probabilities,
            context_history,
            candidates=candidates,
            top_k=tickets_per_draw,
            random_state=random_state + draw_idx,
            objective=objective,
            structure_weight=structure_weight,
            temperature=temperature,
            diversity_weight=diversity_weight,
        )
        random_tickets = _random_tickets(rng, tickets_per_draw)

        rows.extend(_score_ticket_group(target, tickets, f"{model_name}_holdout"))
        rows.extend(_score_ticket_group(target, random_tickets, "random"))
    return pd.DataFrame(rows)


def recommend_models(
    history: pd.DataFrame,
    model_names: list[str],
    train_windows: list[int | None],
    train_until_issue: str,
    tickets_per_draw: int = 3,
    candidates: int = 500,
    max_test_draws: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in model_names:
        for window in train_windows:
            try:
                results = holdout_evaluate(
                    history,
                    model_name=model_name,
                    train_until_issue=train_until_issue,
                    tickets_per_draw=tickets_per_draw,
                    candidates=candidates,
                    max_test_draws=max_test_draws,
                    train_window_draws=window,
                )
                summary = summarize_results(results)
                model_row = summary[summary["strategy"] != "random"].iloc[0].to_dict()
                model_row["model"] = model_name
                model_row["train_window_draws"] = window
                model_row["score"] = _recommendation_score(model_row)
                model_row["error"] = None
                rows.append(model_row)
            except Exception as exc:
                rows.append(
                    {
                        "model": model_name,
                        "train_window_draws": window,
                        "score": -999.0,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("strategy")
    summary = grouped.agg(
        tickets=("strategy", "size"),
        avg_front_hits=("front_hits", "mean"),
        avg_back_hits=("back_hits", "mean"),
        any_prize_rate=("prize_level", lambda x: x.notna().mean()),
        best_prize=("prize_level", lambda x: int(x.dropna().min()) if x.notna().any() else 0),
    )
    for level in range(1, 10):
        summary[f"level_{level}_count"] = grouped["prize_level"].apply(lambda s, lvl=level: int((s == lvl).sum()))
    summary = summary.reset_index()
    random_row = summary[summary["strategy"] == "random"]
    if not random_row.empty:
        random_front = float(random_row.iloc[0]["avg_front_hits"])
        random_back = float(random_row.iloc[0]["avg_back_hits"])
        random_prize = float(random_row.iloc[0]["any_prize_rate"])
        summary["front_lift_vs_random"] = summary["avg_front_hits"] - random_front
        summary["back_lift_vs_random"] = summary["avg_back_hits"] - random_back
        summary["prize_rate_lift_vs_random"] = summary["any_prize_rate"] - random_prize
    return summary


def _backtest_start_idx(history: pd.DataFrame, train_until: str | None, train_until_issue: str | None) -> int:
    if train_until_issue is not None:
        issues = history["issue"].map(_issue_key).to_numpy()
        cutoff = _issue_key(train_until_issue)
        return int(np.searchsorted(issues, cutoff, side="right"))
    if train_until is None:
        raise ValueError("provide either train_until or train_until_issue")
    cutoff = pd.to_datetime(train_until)
    return int(np.searchsorted(history["date"].values, cutoff.to_datetime64(), side="right"))


def _window_history(history: pd.DataFrame, train_window_draws: int | None) -> pd.DataFrame:
    if train_window_draws is None or int(train_window_draws) <= 0 or int(train_window_draws) >= len(history):
        return history.reset_index(drop=True)
    return history.tail(int(train_window_draws)).reset_index(drop=True)


def _recommendation_score(row: dict[str, object]) -> float:
    prize_lift = float(row.get("prize_rate_lift_vs_random") or 0.0)
    front_lift = float(row.get("front_lift_vs_random") or 0.0)
    back_lift = float(row.get("back_lift_vs_random") or 0.0)
    return prize_lift * 10 + front_lift + back_lift


def _issue_key(issue: object) -> int:
    text = str(issue).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"invalid issue: {issue!r}")
    return int(digits)


def _score_ticket_group(target: pd.Series, tickets: list, strategy: str) -> list[dict[str, object]]:
    rows = []
    for rank, ticket in enumerate(tickets, start=1):
        front = ticket.front if hasattr(ticket, "front") else ticket[0]
        back = ticket.back if hasattr(ticket, "back") else ticket[1]
        hit = evaluate_ticket(front, back, target)
        rows.append(
            {
                "issue": str(target["issue"]),
                "date": target["date"],
                "strategy": strategy,
                "rank": rank,
                "front": " ".join(f"{n:02d}" for n in front),
                "back": " ".join(f"{n:02d}" for n in back),
                "front_hits": hit.front_hits,
                "back_hits": hit.back_hits,
                "prize_level": hit.prize_level,
            }
        )
    return rows


def _random_tickets(rng: np.random.Generator, count: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    tickets = []
    for _ in range(count):
        front = tuple(sorted(rng.choice(FRONT_NUMBERS, size=FRONT_PICK, replace=False)))
        back = tuple(sorted(rng.choice(BACK_NUMBERS, size=BACK_PICK, replace=False)))
        tickets.append((front, back))
    return tickets

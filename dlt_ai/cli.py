from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .data import fetch_official_history, import_history_file, load_history, save_history
from .evaluate import backtest, holdout_evaluate, recommend_models, summarize_results
from .models import load_model, predict_probabilities, save_model, train_model_bundle
from .optimize import generate_tickets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dlt-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--out", default="data/all_history.csv")

    p_import = sub.add_parser("import")
    p_import.add_argument("source")
    p_import.add_argument("--out", default="data/all_history.csv")
    p_import.add_argument("--sheet", default=0)

    p_train = sub.add_parser("train")
    p_train.add_argument("--history", default="data/all_history.csv")
    p_train.add_argument("--model", default="random_forest")
    p_train.add_argument("--out", default="models/random_forest.joblib")
    p_train.add_argument("--min-history", type=int, default=30)
    p_train.add_argument("--train-window", type=int)

    p_predict = sub.add_parser("predict")
    p_predict.add_argument("--history", default="data/all_history.csv")
    p_predict.add_argument("--model-path", default="models/random_forest.joblib")
    p_predict.add_argument("--tickets", type=int, default=10)
    p_predict.add_argument("--candidates", type=int, default=100_000)
    p_predict.add_argument("--out", default="data/latest_prediction.csv")

    p_backtest = sub.add_parser("backtest")
    p_backtest.add_argument("--history", default="data/all_history.csv")
    p_backtest.add_argument("--model", default="random_forest")
    p_backtest.add_argument("--train-until")
    p_backtest.add_argument("--train-until-issue")
    p_backtest.add_argument("--tickets", type=int, default=10)
    p_backtest.add_argument("--candidates", type=int, default=20_000)
    p_backtest.add_argument("--out", default="reports/backtest.csv")

    p_holdout = sub.add_parser("holdout")
    p_holdout.add_argument("--history", default="data/all_history.csv")
    p_holdout.add_argument("--model", default="logistic")
    p_holdout.add_argument("--train-until-issue", required=True)
    p_holdout.add_argument("--tickets", type=int, default=10)
    p_holdout.add_argument("--candidates", type=int, default=5_000)
    p_holdout.add_argument("--max-test-draws", type=int)
    p_holdout.add_argument("--train-window", type=int)
    p_holdout.add_argument("--out", default="reports/holdout.csv")
    p_holdout.add_argument("--summary-out", default="reports/holdout_summary.csv")

    p_recommend = sub.add_parser("recommend")
    p_recommend.add_argument("--history", default="data/all_history.csv")
    p_recommend.add_argument("--models", default="logistic,hist_gradient_boosting,gradient_boosting")
    p_recommend.add_argument("--windows", default="300,500,1000,all")
    p_recommend.add_argument("--train-until-issue", required=True)
    p_recommend.add_argument("--tickets", type=int, default=3)
    p_recommend.add_argument("--candidates", type=int, default=500)
    p_recommend.add_argument("--max-test-draws", type=int, default=30)
    p_recommend.add_argument("--out", default="reports/recommendations.csv")

    p_exp = sub.add_parser("experiment")
    p_exp.add_argument("--history", default="data/all_history.csv")
    p_exp.add_argument("--model", default="random_forest")
    p_exp.add_argument("--tickets", type=int, default=10)
    p_exp.add_argument("--candidates", type=int, default=100_000)
    p_exp.add_argument("--log", default="data/experiment_log.jsonl")

    args = parser.parse_args(argv)
    if args.command == "fetch":
        history = fetch_official_history()
        save_history(history, args.out)
        print(f"saved {len(history)} draws to {args.out}")
    elif args.command == "import":
        sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
        history = import_history_file(args.source, sheet_name=sheet)
        save_history(history, args.out)
        print(f"imported {len(history)} draws to {args.out}")
    elif args.command == "train":
        history = load_history(args.history)
        bundle = train_model_bundle(history, model_name=args.model, min_history=args.min_history, train_window_draws=args.train_window)
        save_model(bundle, args.out)
        print(json.dumps(_model_summary(bundle, args.out), ensure_ascii=False, indent=2))
    elif args.command == "predict":
        history = load_history(args.history)
        bundle = load_model(args.model_path)
        prediction = predict_probabilities(bundle, history)
        tickets = generate_tickets(prediction, history, candidates=args.candidates, top_k=args.tickets)
        out_df = pd.DataFrame([ticket.as_dict() for ticket in tickets])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False, encoding="utf-8")
        print(out_df.to_string(index=False))
    elif args.command == "backtest":
        history = load_history(args.history)
        results = backtest(
            history,
            model_name=args.model,
            train_until=args.train_until,
            train_until_issue=args.train_until_issue,
            tickets_per_draw=args.tickets,
            candidates=args.candidates,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False, encoding="utf-8")
        print(summarize_results(results).to_string(index=False))
    elif args.command == "holdout":
        history = load_history(args.history)
        results = holdout_evaluate(
            history,
            model_name=args.model,
            train_until_issue=args.train_until_issue,
            tickets_per_draw=args.tickets,
            candidates=args.candidates,
            max_test_draws=args.max_test_draws,
            train_window_draws=args.train_window,
        )
        summary = summarize_results(results)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.out, index=False, encoding="utf-8")
        summary.to_csv(args.summary_out, index=False, encoding="utf-8")
        print(summary.to_string(index=False))
    elif args.command == "recommend":
        history = load_history(args.history)
        model_names = [item.strip() for item in args.models.split(",") if item.strip()]
        windows = [_parse_window(item) for item in args.windows.split(",") if item.strip()]
        recommendations = recommend_models(
            history,
            model_names=model_names,
            train_windows=windows,
            train_until_issue=args.train_until_issue,
            tickets_per_draw=args.tickets,
            candidates=args.candidates,
            max_test_draws=args.max_test_draws,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(args.out, index=False, encoding="utf-8")
        print(recommendations.to_string(index=False))
    elif args.command == "experiment":
        run_experiment(args)


def run_experiment(args: argparse.Namespace) -> None:
    history = load_history(args.history)
    bundle = train_model_bundle(history, model_name=args.model)
    prediction = predict_probabilities(bundle, history)
    tickets = generate_tickets(prediction, history, candidates=args.candidates, top_k=args.tickets)
    record = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "trained_until_issue": str(history.iloc[-1]["issue"]),
        "trained_until_date": str(history.iloc[-1]["date"].date()),
        "tickets": [ticket.as_dict() for ticket in tickets],
        "front_metrics": bundle["front"]["metrics"],
        "back_metrics": bundle["back"]["metrics"],
    }
    path = Path(args.log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def _model_summary(bundle: dict, path: str) -> dict:
    return {
        "saved_to": path,
        "model": bundle["model_name"],
        "front_metrics": bundle["front"]["metrics"],
        "back_metrics": bundle["back"]["metrics"],
        "last_issue": bundle["front"]["last_issue"],
        "train_window_draws": bundle.get("train_window_draws"),
    }


def _parse_window(value: str) -> int | None:
    text = value.strip().lower()
    if text in {"all", "none", "full", "0"}:
        return None
    return int(text)


if __name__ == "__main__":
    main()





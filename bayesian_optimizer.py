def main() -> None:
    raise SystemExit(
        "Bayesian optimization hook is reserved for tuning model and ticket-generation parameters. "
        "Start with dlt_ai.cli backtest, then optimize against out-of-sample metrics."
    )


if __name__ == "__main__":
    main()


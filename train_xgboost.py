from dlt_ai.cli import main

if __name__ == "__main__":
    main(["train", "--model", "xgboost", "--out", "models/xgboost.joblib"])


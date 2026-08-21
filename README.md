# Daletou AI

A serious machine-learning experiment for China Sports Lottery Super Lotto / Da Le Tou.

The goal is not to claim that lottery draws are predictable. The goal is to build a clean experimental pipeline: acquire history, engineer per-number features, train probability models, optimize ticket combinations, and compare every result against a random baseline with time-based backtesting.

> Product positioning: a historical draw-data analysis and probability education tool. It does not sell or purchase lottery tickets, promise prizes, or present historical statistics as reliable forecasts.

## WeChat Mini Program MVP

The repository includes a native WeChat Mini Program in `miniprogram/`. The first public product provides:

- Latest and recent official draw results
- Searchable, paginated draw history
- Number frequency, omission, and hot/cold labels over 30/50/100/300 draws
- Sum, span, odd/even, and three-zone distributions
- Equal-probability random number generation with local favorites
- A high-tier configurable model lab with custom features, objectives, optimization weights, generation, diagnostics, and backtesting
- A separate quick generator that only asks for model, training window, and ticket count
- Selectable model, training window, feature groups, generation objective, and optimization weights
- Time-based holdout backtesting against an equal-size random baseline
- Saved experiment combinations and automatic post-draw review
- Data-source, probability, and service-boundary statements

The model lab reports historical relative scores and experimental results, not objective winning probabilities. Every user configuration can be checked with the same time-isolated holdout setup used by the research console.

### Local Mini Program Development

1. Install dependencies and start the API:

   ```powershell
   C:\Users\18588\anaconda3\python.exe -m pip install -r requirements.txt
   C:\Users\18588\anaconda3\python.exe run_web.py
   ```

2. Import the repository root in WeChat DevTools. `project.config.json` uses the registered development AppID.
3. Enable "Do not verify valid domains" in DevTools. The local API defaults to `http://127.0.0.1:5000`.

### Production Checklist

- Change `miniprogram/utils/config.js` to an HTTPS API URL on a registered domain.
- Add that host to the Mini Program `request` domain allowlist.
- Copy `.env.example` values into the hosting platform's environment settings and replace `DLT_ADMIN_API_KEY`.
- Set `WECHAT_APP_ID` and `WECHAT_APP_SECRET`, then enable `WECHAT_AUTH_ENABLED` in the Mini Program config. Never put the AppSecret in Mini Program code.
- Run `python run_web.py`; production mode uses Waitress instead of Flask's development server.
- Put the service behind an HTTPS reverse proxy and persist `data/`, `models/`, and `reports/`.
- Schedule data refreshes after draw times. Mini Program launches still check freshness, while the API uses a ten-minute cooldown and incremental recent-page fetches.
- Add privacy guidelines, terms, data-source details, a complaint channel, and operator identity.

Health endpoints:

```text
GET /api/v1/health
GET /api/v1/ready
```

The backend uses WeChat code-to-session authentication, SQLite WAL mode, bounded model-job
execution, restart recovery, per-user data isolation, rotating logs, and server-side favorite synchronization. Expensive legacy
research endpoints require `X-Admin-Key` in production. Create a consistent online backup
while the service is running with:

```bash
python -m dlt_ai.maintenance backup --out backups
python -m dlt_ai.maintenance prune
```

Container deployment is available through the included `Dockerfile`. Mount the three
persistent directories and set `DLT_HOST=0.0.0.0`.

The current build deliberately has no payment or entitlement flow. After approval and retention validation, membership can unlock configurable model experiments, advanced statistics, export, cloud favorites, ad removal, and research reports. Products must be described as historical data and model experiment tools, never as improved winning odds, guaranteed number predictions, ticket purchasing, group buying, or betting.

## Structure

```text
data/                         history, predictions, experiment logs
models/                       trained model bundles
reports/                      backtest reports
notebooks/dashboard.ipynb     dashboard starter notebook
dlt_ai/                       core Python package
scripts/                      PowerShell helpers
tests/                        regression tests
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional models:

```bash
pip install xgboost lightgbm tensorflow shap
```

## Data Ingestion

The preferred first path is manual download from a provincial lottery website, such as the Jiangsu Sports Lottery draw-number table download. Save the downloaded file locally, then import it:

```bash
python -m dlt_ai.cli import path\to\downloaded_table.xlsx --out data/all_history.csv
```

Supported import formats:

```text
.csv, .txt, .xlsx, .xls, .html, .htm
```

The importer accepts either canonical columns:

```text
issue,date,front1,front2,front3,front4,front5,back1,back2
```

or common website-table shapes such as:

```text
issue/date/draw numbers in one combined column
issue/date/front-zone/back-zone columns
Chinese column names like issue number, draw date, draw numbers, front zone, back zone
```


For headerless number-only tables, the importer preserves draw order and issue numbers. Exact draw dates are not required for training or issue-based backtesting.
The official web API fetcher is still available, but it may be blocked by anti-scraping rules in some networks:

```bash
python -m dlt_ai.cli fetch --out data/all_history.csv
```

## Train

```bash
python -m dlt_ai.cli train --model random_forest --history data/all_history.csv --out models/random_forest.joblib
```

## Predict

```bash
python -m dlt_ai.cli predict --model-path models/random_forest.joblib --history data/all_history.csv --tickets 10
```

## Backtest

```bash
python -m dlt_ai.cli backtest --model random_forest --history data/all_history.csv --train-until-issue 24000 --out reports/backtest_after_24000.csv
```

## Holdout Evaluation

Train once on all draws up to an issue, then evaluate later draws against the same number of random tickets:

```bash
python -m dlt_ai.cli holdout --model logistic --history data/all_history.csv --train-until-issue 25000 --tickets 5 --candidates 1000 --max-test-draws 50
```

The command writes both detailed ticket-level results and a summary with lift versus random:

```text
reports/holdout.csv
reports/holdout_summary.csv
```

## Long-Running Experiment

After each draw, update history and log the next prediction:

```bash
python -m dlt_ai.cli experiment --history data/all_history.csv --model random_forest --tickets 10
```

## Web App

Run the local experiment dashboard:

```bash
python run_web.py
```

Then open:

```text
http://127.0.0.1:5000
```

On every page load the app calls the refresh endpoint, attempts to fetch official latest draws, merges new rows into `data/all_history.csv`, evaluates saved predictions whose target draw is now known, and keeps experiment records in `data/app.db`.

## Method

The model does not directly predict one ticket. It expands every draw into per-number binary classification targets:

```text
P(front 01 appears), ..., P(front 35 appears)
P(back 01 appears), ..., P(back 12 appears)
```

Features include omission, rolling counts, rolling rates, EMA, gap statistics, recent streaks, neighbor indicators, mirror indicators, prime/tail/odd/small/zone properties, and previous-draw aggregate statistics.

Ticket generation then combines model probabilities with historical-shape constraints such as sum, span, odd/even count, and front-zone distribution.

## Reality Check

Lottery draws should be treated as near-random independent events. A negative result is still a useful data-science result if it comes from clean out-of-sample evaluation.


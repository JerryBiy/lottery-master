from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import BACK_MAX, FRONT_MAX, HISTORY_COLUMNS, OFFICIAL_HISTORY_URL

PROVINCIAL_HISTORY_URL = "https://www.gstc.org.cn/workapi/prize_history_list"


NOISE_COL_KEYWORDS = (
    "\u6ce8",
    "\u6c60",
    "\u9500\u552e",
    "\u6eda\u5b58",
    "\u5956\u91d1",
    "\u4e00\u7b49\u5956",
    "\u4e8c\u7b49\u5956",
    "\u4e09\u7b49\u5956",
    "\u56db\u7b49\u5956",
    "\u4e94\u7b49\u5956",
    "\u516d\u7b49\u5956",
    "\u4e03\u7b49\u5956",
    "\u516b\u7b49\u5956",
    "\u4e5d\u7b49\u5956",
)


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical draw history sorted by issue."""
    rename_map = {
        "draw_num": "issue",
        "draw_date": "date",
        "lotteryDrawNum": "issue",
        "lotteryDrawTime": "date",
    }
    df = df.rename(columns=rename_map).copy()

    missing = [col for col in HISTORY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")

    df = df[HISTORY_COLUMNS].copy()
    df["issue"] = df["issue"].map(_stringify_issue)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in HISTORY_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    if df["date"].isna().any():
        raise ValueError("history contains invalid date values")

    _validate_ranges(df)
    return df.drop_duplicates("issue").sort_values(["date", "issue"]).reset_index(drop=True)


def load_history(path: str | Path) -> pd.DataFrame:
    return normalize_history(pd.read_csv(path, dtype={"issue": str}))


def import_history_file(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Import a downloaded CSV/Excel/HTML lottery table and canonicalize it."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        try:
            return canonicalize_loose_history(pd.read_csv(path, dtype=str, header=None))
        except ValueError:
            return canonicalize_loose_history(pd.read_csv(path, dtype=str))
    if suffix in {".xlsx", ".xls"}:
        try:
            return canonicalize_loose_history(pd.read_excel(path, sheet_name=sheet_name, dtype=str, header=None))
        except ValueError:
            return canonicalize_loose_history(pd.read_excel(path, sheet_name=sheet_name, dtype=str))
    if suffix in {".html", ".htm"}:
        tables = pd.read_html(path)
        candidates = []
        for table in tables:
            try:
                candidates.append(canonicalize_loose_history(table))
            except ValueError:
                continue
        if not candidates:
            raise ValueError(f"no recognizable lottery table found in {path}")
        return max(candidates, key=len)
    raise ValueError(f"unsupported history file type: {suffix}")


def canonicalize_loose_history(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort parser for provincial website CSV/Excel/HTML tables."""
    df = df.copy().dropna(how="all")
    df.columns = [_clean_col(col) for col in df.columns]

    direct = _direct_column_mapping(df)
    if direct is not None:
        return normalize_history(direct)

    positional = _canonicalize_positional_table(df)
    if positional is not None:
        return normalize_history(positional)

    issue_col = _find_first_col(df.columns, ["\u5f00\u5956\u671f\u53f7", "\u671f\u53f7", "issue", "drawnum", "draw_num"])
    date_col = _find_first_col(df.columns, ["\u5f00\u5956\u65e5\u671f", "\u5f00\u5956\u65f6\u95f4", "\u65e5\u671f", "\u65f6\u95f4", "date", "drawdate"])
    if issue_col is None:
        raise ValueError("could not identify issue column")
    if date_col is None:
        raise ValueError("could not identify draw date column")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        issue = _stringify_issue(row.get(issue_col))
        date = row.get(date_col)
        front, back = _extract_numbers_from_row(row, issue_col=issue_col, date_col=date_col)
        if not issue or front is None or back is None:
            continue
        rows.append(
            {
                "issue": issue,
                "date": date,
                "front1": front[0],
                "front2": front[1],
                "front3": front[2],
                "front4": front[3],
                "front5": front[4],
                "back1": back[0],
                "back2": back[1],
            }
        )

    if not rows:
        raise ValueError("no recognizable draw rows found")
    return normalize_history(pd.DataFrame(rows))


def save_history(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = normalize_history(df)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    out.to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(path)


def fetch_official_history(page_size: int = 100, sleep: float = 0.15, max_pages: int = 200) -> pd.DataFrame:
    """Fetch draw history from the public China Sports Lottery endpoint."""
    rows: list[dict[str, Any]] = []
    page = 1
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.lottery.gov.cn/",
        "Origin": "https://www.lottery.gov.cn",
    }

    while page <= max_pages:
        params = {
            "gameNo": 85,
            "provinceId": 0,
            "pageSize": page_size,
            "isVerify": 1,
            "pageNo": page,
        }
        response = requests.get(OFFICIAL_HISTORY_URL, params=params, headers=headers, timeout=20)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                "official endpoint rejected the request. "
                "If this persists, download the official history manually and save it as CSV."
            ) from exc
        payload = response.json()
        records = _extract_records(payload)
        if not records:
            break

        for record in records:
            rows.append(_parse_official_record(record))

        total_pages = _extract_total_pages(payload)
        if total_pages is not None and page >= total_pages:
            break
        page += 1
        time.sleep(sleep)

    if not rows:
        raise RuntimeError("official endpoint returned no draw rows")
    return normalize_history(pd.DataFrame(rows))


def fetch_provincial_history(page_size: int = 100, max_pages: int = 2) -> pd.DataFrame:
    """Fetch recent draws from the Gansu Sports Lottery official API."""
    rows: list[dict[str, Any]] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.gstc.org.cn/wanfa/dlt",
    }
    for page in range(1, max_pages + 1):
        response = requests.post(
            PROVINCIAL_HISTORY_URL,
            json={"type_id": "001", "page_no": page, "page_size": str(page_size)},
            headers=headers,
            timeout=(5, 15),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "000000":
            raise RuntimeError(f"provincial endpoint returned {payload.get('code')}: {payload.get('message')}")
        records = payload.get("data", {}).get("prize_history_list", [])
        if not records:
            break
        rows.extend(_parse_provincial_record(record) for record in records)
        if len(records) < page_size:
            break

    if not rows:
        raise RuntimeError("provincial endpoint returned no draw rows")
    history = normalize_history(pd.DataFrame(rows))
    history.attrs["source"] = "甘肃省体育彩票管理中心"
    return history


def fetch_recent_official_history(page_size: int = 100) -> pd.DataFrame:
    """Fetch recent draws with an official provincial source and national fallback."""
    errors = []
    try:
        return fetch_provincial_history(page_size=page_size, max_pages=2)
    except Exception as exc:
        errors.append(f"provincial: {exc}")

    try:
        history = fetch_official_history(page_size=min(page_size, 100), max_pages=2)
        history.attrs["source"] = "中国体育彩票"
        return history
    except Exception as exc:
        errors.append(f"national: {exc}")
    raise RuntimeError("; ".join(errors))


def _canonicalize_positional_table(df: pd.DataFrame) -> pd.DataFrame | None:
    if df.shape[1] < 8:
        return None
    candidate = df.iloc[:, :8].copy().dropna(how="all")
    candidate.columns = ["issue", "front1", "front2", "front3", "front4", "front5", "back1", "back2"]
    for col in candidate.columns:
        candidate[col] = candidate[col].map(_stringify_issue)
    numeric = candidate.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    if numeric.isna().any().any():
        return None
    out = candidate.copy()
    for col in candidate.columns[1:]:
        out[col] = numeric[col].astype(int)
    out = out.sort_values("issue", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    out["date"] = pd.date_range("2000-01-01", periods=len(out), freq="D")
    return out[HISTORY_COLUMNS].copy()

def _direct_column_mapping(df: pd.DataFrame) -> pd.DataFrame | None:
    lower_map = {_clean_key(col): col for col in df.columns}
    if all(col in df.columns for col in HISTORY_COLUMNS):
        return df[HISTORY_COLUMNS].copy()

    mapping: dict[str, str] = {}
    aliases = {
        "issue": ["issue", "drawnum", "draw_num", "lotterydrawnum", "\u671f\u53f7", "\u5f00\u5956\u671f\u53f7"],
        "date": ["date", "drawdate", "draw_date", "lotterydrawtime", "\u5f00\u5956\u65e5\u671f", "\u5f00\u5956\u65f6\u95f4", "\u65e5\u671f"],
        "front1": ["front1", "\u524d\u533a1", "\u524d\u4e00", "\u7ea2\u74031"],
        "front2": ["front2", "\u524d\u533a2", "\u524d\u4e8c", "\u7ea2\u74032"],
        "front3": ["front3", "\u524d\u533a3", "\u524d\u4e09", "\u7ea2\u74033"],
        "front4": ["front4", "\u524d\u533a4", "\u524d\u56db", "\u7ea2\u74034"],
        "front5": ["front5", "\u524d\u533a5", "\u524d\u4e94", "\u7ea2\u74035"],
        "back1": ["back1", "\u540e\u533a1", "\u540e\u4e00", "\u84dd\u74031"],
        "back2": ["back2", "\u540e\u533a2", "\u540e\u4e8c", "\u84dd\u74032"],
    }
    for target, names in aliases.items():
        for name in names:
            key = _clean_key(name)
            if key in lower_map:
                mapping[target] = lower_map[key]
                break
    if all(col in mapping for col in HISTORY_COLUMNS):
        return df.rename(columns={source: target for target, source in mapping.items()})[HISTORY_COLUMNS].copy()
    return None


def _extract_numbers_from_row(row: pd.Series, issue_col: str, date_col: str) -> tuple[list[int] | None, list[int] | None]:
    front_text = _join_candidate_values(row, ["\u524d\u533a", "\u524d\u533a\u53f7\u7801", "\u524d"])
    back_text = _join_candidate_values(row, ["\u540e\u533a", "\u540e\u533a\u53f7\u7801", "\u540e"])
    front_numbers = _numbers_from_text(front_text)
    back_numbers = _numbers_from_text(back_text)
    if len(front_numbers) >= 5 and len(back_numbers) >= 2:
        return sorted(front_numbers[:5]), sorted(back_numbers[:2])

    draw_text = _join_candidate_values(row, ["\u5f00\u5956\u53f7\u7801", "\u4e2d\u5956\u53f7\u7801", "\u5f00\u5956\u53f7", "\u53f7\u7801", "\u7ed3\u679c"])
    draw_numbers = _numbers_from_text(draw_text)
    if len(draw_numbers) >= 7:
        return sorted(draw_numbers[:5]), sorted(draw_numbers[5:7])

    values = []
    for col, value in row.items():
        if col in {issue_col, date_col} or _is_noise_col(col):
            continue
        values.extend(_numbers_from_text(value))
    if len(values) >= 7:
        return sorted(values[:5]), sorted(values[5:7])
    return None, None


def _join_candidate_values(row: pd.Series, keywords: list[str]) -> str:
    values = []
    for col, value in row.items():
        col_key = _clean_key(col)
        if any(_clean_key(keyword) in col_key for keyword in keywords) and not _is_noise_col(col):
            values.append(str(value))
    return " ".join(values)


def _numbers_from_text(value: Any) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value)
    return [int(token) for token in re.findall(r"(?<!\d)\d{1,2}(?!\d)", text)]


def _clean_col(col: Any) -> str:
    return re.sub(r"\s+", "", str(col)).strip()


def _clean_key(value: Any) -> str:
    return re.sub(r"[\s_\-:/\uff08\uff09()]+", "", str(value)).lower()


def _find_first_col(columns: pd.Index, names: list[str]) -> str | None:
    cleaned = [(_clean_key(col), col) for col in columns]
    for name in names:
        needle = _clean_key(name)
        for cleaned_col, original in cleaned:
            if needle in cleaned_col:
                return str(original)
    return None


def _is_noise_col(col: Any) -> bool:
    return any(keyword in str(col) for keyword in NOISE_COL_KEYWORDS)


def _stringify_issue(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _extract_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("value", payload)
    if isinstance(value, dict):
        for key in ("list", "records", "rows"):
            records = value.get(key)
            if isinstance(records, list):
                return records
    if isinstance(value, list):
        return value
    return []


def _extract_total_pages(payload: dict[str, Any]) -> int | None:
    value = payload.get("value", payload)
    if not isinstance(value, dict):
        return None
    for key in ("pages", "totalPages", "pageCount"):
        if key in value:
            return int(value[key])
    total = value.get("total")
    page_size = value.get("pageSize")
    if total and page_size:
        return int((int(total) + int(page_size) - 1) / int(page_size))
    return None


def _parse_official_record(record: dict[str, Any]) -> dict[str, Any]:
    issue = str(record.get("lotteryDrawNum") or record.get("drawNum") or record.get("issue"))
    date = record.get("lotteryDrawTime") or record.get("drawTime") or record.get("date")
    raw_result = record.get("lotteryDrawResult") or record.get("drawResult") or record.get("result")
    if not raw_result:
        raise ValueError(f"record has no draw result: {record}")

    parts = [int(x) for x in str(raw_result).replace(",", " ").split()]
    if len(parts) != 7:
        raise ValueError(f"expected 7 numbers for issue {issue}, got {raw_result!r}")
    front = sorted(parts[:5])
    back = sorted(parts[5:])
    return {
        "issue": issue,
        "date": date,
        "front1": front[0],
        "front2": front[1],
        "front3": front[2],
        "front4": front[3],
        "front5": front[4],
        "back1": back[0],
        "back2": back[1],
    }


def _parse_provincial_record(record: dict[str, Any]) -> dict[str, Any]:
    numbers = [int(number) for number in record.get("prize_num", [])]
    if len(numbers) != 7:
        raise ValueError(f"expected 7 numbers in provincial record: {record}")
    front = sorted(numbers[:5])
    back = sorted(numbers[5:])
    return {
        "issue": str(record["issue_num"]),
        "date": record["prize_time"],
        "front1": front[0],
        "front2": front[1],
        "front3": front[2],
        "front4": front[3],
        "front5": front[4],
        "back1": back[0],
        "back2": back[1],
    }


def _validate_ranges(df: pd.DataFrame) -> None:
    front_cols = [f"front{i}" for i in range(1, 6)]
    back_cols = [f"back{i}" for i in range(1, 3)]
    for col in front_cols:
        bad = ~df[col].between(1, FRONT_MAX)
        if bad.any():
            raise ValueError(f"{col} contains values outside 1..{FRONT_MAX}")
    for col in back_cols:
        bad = ~df[col].between(1, BACK_MAX)
        if bad.any():
            raise ValueError(f"{col} contains values outside 1..{BACK_MAX}")
    if df[front_cols].apply(lambda row: len(set(row)) != 5, axis=1).any():
        raise ValueError("front numbers must be unique within each draw")
    if df[back_cols].apply(lambda row: len(set(row)) != 2, axis=1).any():
        raise ValueError("back numbers must be unique within each draw")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/all_history.csv")
    parser.add_argument("--page-size", type=int, default=100)
    args = parser.parse_args(argv)
    history = fetch_official_history(page_size=args.page_size)
    save_history(history, args.out)
    print(f"saved {len(history)} draws to {args.out}")


if __name__ == "__main__":
    main()



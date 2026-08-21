import pandas as pd

import pytest
from dataclasses import replace

from dlt_ai.data import canonicalize_loose_history, normalize_history, save_history
from dlt_ai.evaluate import prize_level
from dlt_ai.features import build_prediction_frame, build_training_frame
from dlt_ai.models import select_feature_columns, train_model_bundle
from dlt_ai.optimize import generate_tickets
from dlt_ai.product import (
    distribution_statistics,
    generate_random_tickets,
    home_payload,
    number_statistics,
    random_candidate_counts,
)
from dlt_ai.webapp import _evaluate_favorite_groups
import dlt_ai.webapp as webapp


def sample_history(draws: int = 45) -> pd.DataFrame:
    rows = []
    for idx in range(draws):
        front = sorted((((idx * 3 + j * 5) % 35) + 1 for j in range(5)))
        if len(set(front)) < 5:
            front = sorted({*front, ((idx + 17) % 35) + 1})[:5]
        back = sorted((((idx * 2 + j * 3) % 12) + 1 for j in range(2)))
        rows.append(
            {
                "issue": f"{idx + 1:05d}",
                "date": f"2020-01-{(idx % 28) + 1:02d}",
                "front1": front[0],
                "front2": front[1],
                "front3": front[2],
                "front4": front[3],
                "front5": front[4],
                "back1": back[0],
                "back2": back[1],
            }
        )
    return normalize_history(pd.DataFrame(rows))


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    history_path = tmp_path / "all_history.csv"
    database_path = tmp_path / "app.db"
    save_history(sample_history(60), history_path)
    monkeypatch.setattr(webapp, "HISTORY_PATH", history_path)
    monkeypatch.setattr(webapp, "DB_PATH", database_path)
    app = webapp.create_app(start_scheduler=False)
    app.config["TESTING"] = True
    return app.test_client()


def test_feature_shapes():
    history = sample_history()
    X, meta = build_prediction_frame(history, "front")
    assert len(X) == 35
    assert len(meta) == 35
    frame = build_training_frame(history, "back", min_history=30)
    assert len(frame.X) == (len(history) - 30) * 12
    assert len(frame.y) == len(frame.X)


def test_prize_rules():
    assert prize_level(5, 2) == 1
    assert prize_level(5, 1) == 2
    assert prize_level(4, 0) == 7
    assert prize_level(0, 1) is None


def test_loose_import_from_combined_numbers():
    raw = pd.DataFrame(
        {
            "\u5f00\u5956\u671f\u53f7": ["25001", "25002"],
            "\u5f00\u5956\u65e5\u671f": ["2025-01-01", "2025-01-04"],
            "\u5f00\u5956\u53f7\u7801": ["01 05 12 22 35 + 03 11", "02 06 13 23 34 + 04 12"],
            "\u9500\u552e\u989d": ["100", "200"],
        }
    )
    history = canonicalize_loose_history(raw)
    assert list(history.columns) == ["issue", "date", "front1", "front2", "front3", "front4", "front5", "back1", "back2"]
    assert history.iloc[0]["front5"] == 35
    assert history.iloc[1]["back2"] == 12


def test_loose_import_from_front_back_columns():
    raw = pd.DataFrame(
        {
            "\u671f\u53f7": ["25003"],
            "\u65e5\u671f": ["2025-01-06"],
            "\u524d\u533a": ["03 07 14 24 33"],
            "\u540e\u533a": ["01 09"],
        }
    )
    history = canonicalize_loose_history(raw)
    assert history.iloc[0]["front1"] == 3
    assert history.iloc[0]["back2"] == 9


def test_loose_import_from_headerless_8_column_table():
    raw = pd.DataFrame(
        [
            ["26073", "04", "10", "22", "23", "33", "02", "12"],
            ["26072", "01", "13", "26", "29", "30", "09", "11"],
        ]
    )
    history = canonicalize_loose_history(raw)
    assert list(history["issue"]) == ["26072", "26073"]
    assert history.iloc[-1]["front1"] == 4
    assert history.iloc[-1]["back2"] == 12


def test_product_statistics_payloads():
    history = sample_history(120)
    home = home_payload(history)
    numbers = number_statistics(history, 50)
    distributions = distribution_statistics(history, 50)

    assert home["draw_count"] == 120
    assert len(home["recent"]) == 6
    assert len(numbers["front"]) == 35
    assert len(numbers["back"]) == 12
    assert numbers["window"] == 50
    assert sum(item["count"] for item in numbers["front"]) == 50 * 5
    assert sum(item["count"] for item in distributions["odd_distribution"]) == 50
    assert "average_back_sum" in home["quick_stats"]
    assert distributions["back_sum"]["minimum"] >= 3
    assert distributions["front_sum"]["level"] in {"low", "normal", "high"}
    assert sum(item["count"] for item in distributions["front_sum_histogram"]) == 50
    assert sum(item["count"] for item in distributions["back_sum_histogram"]) == 50
    assert len(distributions["pattern_metrics"]) == 5
    assert len(distributions["recent_series"]) == 20
    assert len(distributions["omission_matrix"]["front"]) == 15
    assert len(distributions["omission_matrix"]["front"][0]["cells"]) == 35
    assert len(distributions["omission_matrix"]["back"][0]["cells"]) == 12
    assert len(distributions["tail_frequency"]["front"]) == 10
    assert len(distributions["position_statistics"]) == 5
    assert len(distributions["gap_statistics"]) == 4
    assert len(distributions["top_pairs"]) == 10
    assert len(distributions["five_zone_distribution"]) <= 8
    assert set(distributions["ratio_distributions"]) == {
        "big_small",
        "prime_composite",
        "back_odd_even",
        "back_big_small",
        "route_012",
    }


def test_random_tickets_are_valid_and_bounded():
    tickets = generate_random_tickets(100, seed=42)
    assert len(tickets) == 20
    for ticket in tickets:
        assert len(ticket["front"]) == len(set(ticket["front"])) == 5
        assert len(ticket["back"]) == len(set(ticket["back"])) == 2
        assert min(ticket["front"]) >= 1 and max(ticket["front"]) <= 35
        assert min(ticket["back"]) >= 1 and max(ticket["back"]) <= 12


def test_custom_random_tickets_follow_every_constraint():
    constraints = {
        "frontRequired": [3, 18],
        "frontExcluded": [1, 2, 35],
        "backRequired": [7],
        "backExcluded": [1, 2],
        "frontOddCount": 2,
        "frontBigCount": 3,
        "backOddCount": 1,
        "frontSumMin": 70,
        "frontSumMax": 115,
        "consecutive": "avoid",
        "uniqueTails": True,
        "zoneMode": "cover",
    }
    tickets = generate_random_tickets(20, seed=7, constraints=constraints)

    assert len(tickets) == 20
    assert len({(tuple(ticket["front"]), tuple(ticket["back"])) for ticket in tickets}) == 20
    for ticket in tickets:
        front = ticket["front"]
        back = ticket["back"]
        assert {3, 18}.issubset(front)
        assert not {1, 2, 35}.intersection(front)
        assert 7 in back and not {1, 2}.intersection(back)
        assert sum(number % 2 for number in front) == 2
        assert sum(number >= 18 for number in front) == 3
        assert sum(number % 2 for number in back) == 1
        assert 70 <= sum(front) <= 115
        assert all(right - left != 1 for left, right in zip(front, front[1:]))
        assert len({number % 10 for number in front}) == 5
        assert any(number <= 12 for number in front)
        assert any(13 <= number <= 24 for number in front)
        assert any(number >= 25 for number in front)


def test_custom_random_rejects_conflicts_and_impossible_rules():
    with pytest.raises(ValueError, match="不能同时必选和排除"):
        generate_random_tickets(
            1,
            constraints={"frontRequired": [5], "frontExcluded": [5]},
        )
    with pytest.raises(ValueError, match="没有可用组合"):
        generate_random_tickets(
            1,
            constraints={
                "frontRequired": [1, 2, 3, 4, 5],
                "frontOddCount": 0,
            },
        )


def test_random_candidate_counts_match_unrestricted_space():
    counts = random_candidate_counts()
    assert counts["front"] == 324632
    assert counts["back"] == 66
    assert counts["tickets"] == counts["front"] * counts["back"]


def test_feature_groups_change_the_actual_training_columns():
    history = sample_history(50)
    trend_bundle = train_model_bundle(
        history,
        model_name="logistic",
        feature_groups=["trend"],
        model_preset="conservative",
    )
    mixed_bundle = train_model_bundle(
        history,
        model_name="logistic",
        feature_groups=["heat", "properties"],
        model_preset="conservative",
    )

    assert trend_bundle["front"]["feature_columns"] != mixed_bundle["front"]["feature_columns"]
    assert all(
        column.startswith(("count_", "rate_", "ema_"))
        or column in {"frequency", "last5_trend", "appeared_last", "appeared_previous"}
        for column in trend_bundle["front"]["feature_columns"]
    )
    with pytest.raises(ValueError):
        select_feature_columns(trend_bundle["front"]["feature_columns"], ["unknown"])


def test_generation_objectives_return_valid_ticket_sets():
    history = sample_history(60)
    probabilities = pd.DataFrame(
        [
            *({"zone": "front", "number": number, "probability": 0.05 + number / 1000} for number in range(1, 36)),
            *({"zone": "back", "number": number, "probability": 0.12 + number / 1000} for number in range(1, 13)),
        ]
    )
    for objective in ("balanced", "score", "structure", "coverage"):
        tickets = generate_tickets(
            probabilities,
            history,
            candidates=300,
            top_k=5,
            objective=objective,
            random_state=7,
        )
        assert len(tickets) == 5
        assert all(len(set(ticket.front)) == 5 and len(set(ticket.back)) == 2 for ticket in tickets)


def test_favorite_evaluation_uses_the_next_draw():
    history = sample_history(45)
    target = history.iloc[43]
    group = {
        "id": "saved-1",
        "generatedAfterIssue": str(history.iloc[42]["issue"]),
        "tickets": [
            {
                "front": [int(target[f"front{i}"]) for i in range(1, 6)],
                "back": [int(target[f"back{i}"]) for i in range(1, 3)],
            }
        ],
    }
    result = _evaluate_favorite_groups(history, [group])[0]
    assert result["status"] == "evaluated"
    assert result["target"]["issue"] == str(target["issue"])
    assert result["tickets"][0]["prize_label"] == "一等奖"


def test_backend_health_and_validation_errors(api_client):
    health = api_client.get("/api/v1/health")
    ready = api_client.get("/api/v1/ready")
    invalid = api_client.get("/api/v1/statistics/numbers?window=invalid")

    assert health.status_code == 200
    assert health.get_json()["ok"] is True
    assert ready.status_code == 200
    assert ready.get_json()["checks"]["draw_count"] == 60
    assert invalid.status_code == 400
    assert invalid.get_json()["request_id"]


def test_custom_random_api_returns_space_and_clear_no_solution_error(api_client):
    response = api_client.post(
        "/api/v1/random",
        json={
            "count": 5,
            "constraints": {
                "frontRequired": [8],
                "frontExcluded": [1, 2, 3],
                "backRequired": [6],
                "consecutive": "avoid",
                "zoneMode": "cover",
            },
        },
    )
    impossible = api_client.post(
        "/api/v1/random",
        json={
            "count": 1,
            "constraints": {
                "frontRequired": [1, 2, 3, 4, 5],
                "frontOddCount": 0,
            },
        },
    )

    assert response.status_code == 200
    assert response.get_json()["customized"] is True
    assert response.get_json()["candidate_counts"]["tickets"] > 0
    assert all(8 in ticket["front"] for ticket in response.get_json()["tickets"])
    assert impossible.status_code == 400
    assert "没有可用组合" in impossible.get_json()["error"]


def test_server_favorites_are_isolated_by_client(api_client):
    favorite = {
        "id": "favorite-0001",
        "source": "quick_model",
        "generatedAfterIssue": "00059",
        "tickets": [{"front": [1, 2, 3, 4, 5], "back": [1, 2]}],
    }
    owner_a = {"X-Client-Id": "client-owner-a"}
    owner_b = {"X-Client-Id": "client-owner-b"}

    created = api_client.post("/api/v1/favorites", json=favorite, headers=owner_a)
    own_rows = api_client.get("/api/v1/favorites", headers=owner_a).get_json()["groups"]
    other_rows = api_client.get("/api/v1/favorites", headers=owner_b).get_json()["groups"]
    deleted = api_client.delete("/api/v1/favorites/favorite-0001", headers=owner_a)

    assert created.status_code == 201
    assert len(own_rows) == 1
    assert other_rows == []
    assert deleted.status_code == 200
    assert api_client.get("/api/v1/favorites", headers=owner_a).get_json()["groups"] == []


def test_generation_history_is_private(api_client):
    owner_a = "client-owner-a"
    owner_b = "client-owner-b"
    now = "2026-07-30T20:00:00"
    with webapp.connect() as db:
        db.execute(
            """
            insert into lab_jobs
            (id, job_type, status, progress, message, params_json, result_json,
             error, created_at, updated_at, owner_key)
            values (?, 'generate', 'completed', 100, 'done', '{}', '{}',
                    null, ?, ?, ?)
            """,
            ("job-private-0001", now, now, owner_a),
        )

    hidden = api_client.get("/api/v1/lab/jobs/job-private-0001", headers={"X-Client-Id": owner_b})
    hidden_delete = api_client.post(
        "/api/v1/lab/history/job-private-0001/delete",
        headers={"X-Client-Id": owner_b},
    )
    own_delete = api_client.post(
        "/api/v1/lab/history/job-private-0001/delete",
        headers={"X-Client-Id": owner_a},
    )

    assert hidden.status_code == 404
    assert hidden_delete.status_code == 404
    assert own_delete.status_code == 200


def test_database_uses_wal_and_production_indexes(api_client):
    with webapp.connect() as db:
        journal_mode = db.execute("pragma journal_mode").fetchone()[0]
        indexes = {row["name"] for row in db.execute("pragma index_list(lab_jobs)")}
        columns = {row["name"] for row in db.execute("pragma table_info(lab_jobs)")}

    assert journal_mode == "wal"
    assert "idx_lab_jobs_owner_created" in indexes
    assert "owner_key" in columns


def test_production_user_routes_require_login(api_client, monkeypatch):
    monkeypatch.setattr(webapp, "SETTINGS", replace(webapp.SETTINGS, environment="production"))
    response = api_client.get("/api/v1/favorites")

    assert response.status_code == 401
    assert response.get_json()["error"] == "请先登录"


def test_statistics_response_is_cached_per_history_version(api_client, monkeypatch):
    original = webapp.distribution_statistics
    calls = 0

    def counted(history, window):
        nonlocal calls
        calls += 1
        return original(history, window)

    monkeypatch.setattr(webapp, "distribution_statistics", counted)
    first = api_client.get("/api/v1/statistics/distributions?window=100")
    second = api_client.get("/api/v1/statistics/distributions?window=100")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.get_json() == second.get_json()
    assert calls == 1

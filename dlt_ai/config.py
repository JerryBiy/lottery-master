from __future__ import annotations

FRONT_MAX = 35
BACK_MAX = 12
FRONT_PICK = 5
BACK_PICK = 2

FRONT_NUMBERS = tuple(range(1, FRONT_MAX + 1))
BACK_NUMBERS = tuple(range(1, BACK_MAX + 1))

ROLLING_WINDOWS = (5, 10, 20, 50, 100)
EMA_SPANS = (5, 10, 20, 50)

OFFICIAL_HISTORY_URL = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"

HISTORY_COLUMNS = [
    "issue",
    "date",
    "front1",
    "front2",
    "front3",
    "front4",
    "front5",
    "back1",
    "back2",
]


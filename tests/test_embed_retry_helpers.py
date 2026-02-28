from __future__ import annotations

from src.index.embed import BACKOFF_CAP_S, compute_backoff_s, parse_retry_ms


def test_parse_retry_ms() -> None:
    message = "Please try again in 125ms"
    assert parse_retry_ms(message) == 125


def test_backoff_schedule_monotonic() -> None:
    values = [compute_backoff_s(attempt) for attempt in range(1, 10)]
    assert values == sorted(values)
    assert values[-1] <= BACKOFF_CAP_S
    assert values[-1] == BACKOFF_CAP_S

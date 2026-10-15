"""Small pure helpers for the low-rate, five-second capture quality window."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


WINDOW_NS = 5_000_000_000


def summarize_quality_window(
    frame_receipts: Iterable[int],
    samples: Iterable[Mapping[str, Any]],
    *,
    now_ns: int,
    reader_started_ns: int | None,
    last_valid_ns: int | None,
    window_ns: int = WINDOW_NS,
) -> dict[str, Any]:
    """Summarize real successful-read times and processed sample statuses.

    The frame-rate denominator is the part of the five-second window for which
    the reader has actually been running; it is not the span between frames.
    """
    if window_ns < 1:
        raise ValueError("window_ns must be positive")
    if reader_started_ns is None or now_ns < reader_started_ns:
        return _empty_quality()
    observed_start_ns = max(reader_started_ns, now_ns - window_ns)
    observed_seconds = max(0.0, (now_ns - observed_start_ns) / 1_000_000_000.0)
    if observed_seconds <= 0:
        return _empty_quality()

    lower_ns = now_ns - window_ns
    receipts = [
        value for value in frame_receipts
        if isinstance(value, int)
        and not isinstance(value, bool)
        and observed_start_ns <= value <= now_ns
    ]
    window_samples = [
        sample for sample in samples
        if (timestamp := _sample_time(sample)) is not None and observed_start_ns <= timestamp <= now_ns
    ]
    statuses = [str(sample.get("sample_status") or "unknown") for sample in window_samples]
    warmup_count = sum(status == "warmup" for status in statuses)
    non_warmup_count = len(statuses) - warmup_count
    valid_count = sum(
        bool(sample.get("sample_accepted"))
        for sample in window_samples
        if str(sample.get("sample_status") or "unknown") != "warmup"
    )
    invalid_count = non_warmup_count - valid_count

    last_window_sample = max(window_samples, key=_sample_time) if window_samples else None
    head_state = _channel_state(last_window_sample, "head")
    body_state = _channel_state(last_window_sample, "body")
    tracking_lost = _tracking_lost_state(window_samples, now_ns)
    no_valid_start = last_valid_ns if last_valid_ns is not None else reader_started_ns
    continuous_no_valid_seconds = max(
        0.0,
        (now_ns - no_valid_start) / 1_000_000_000.0,
    )

    return {
        "schema": "localgaze.capture-quality/1",
        "window_seconds": window_ns / 1_000_000_000.0,
        "observed_seconds": observed_seconds,
        "window_start_monotonic_ns": str(observed_start_ns),
        "observed_frame_count": len(receipts),
        "read_frame_rate_hz": len(receipts) / observed_seconds if receipts else None,
        "processed_sample_count": len(window_samples),
        "effective_sample_rate_hz": valid_count / observed_seconds if window_samples else None,
        "warmup_count": warmup_count,
        "non_warmup_count": non_warmup_count,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_ratio": valid_count / non_warmup_count if non_warmup_count else None,
        "invalid_ratio": invalid_count / non_warmup_count if non_warmup_count else None,
        "continuous_no_valid_sample_seconds": continuous_no_valid_seconds,
        "head_state": head_state,
        "body_state": body_state,
        "tracking_lost_seconds": tracking_lost,
        "source": "successful VideoCapture.read receipts and processed capture records; not exposure timing",
    }


def _empty_quality() -> dict[str, Any]:
    return {
        "schema": "localgaze.capture-quality/1",
        "window_seconds": 5.0,
        "observed_seconds": None,
        "window_start_monotonic_ns": None,
        "observed_frame_count": 0,
        "read_frame_rate_hz": None,
        "processed_sample_count": 0,
        "effective_sample_rate_hz": None,
        "warmup_count": 0,
        "non_warmup_count": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "valid_ratio": None,
        "invalid_ratio": None,
        "continuous_no_valid_sample_seconds": None,
        "head_state": {"status": "NA", "reason": "no_samples"},
        "body_state": {"status": "NA", "reason": "no_samples"},
        "tracking_lost_seconds": None,
        "source": "successful VideoCapture.read receipts and processed capture records; not exposure timing",
    }


def _sample_time(sample: Mapping[str, Any]) -> int | None:
    value = sample.get("received_monotonic_ns")
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed >= 0 else None


def _channel_state(sample: Mapping[str, Any] | None, channel: str) -> dict[str, Any]:
    if sample is None:
        return {"status": "NA", "reason": "no_samples"}
    available = sample.get(f"motion_{channel}_available")
    valid = sample.get(f"motion_{channel}_valid")
    if available is None:
        return {"status": "NA", "reason": "source_not_provided"}
    if not bool(available):
        return {"status": "unavailable", "reason": sample.get(f"motion_{channel}_reason")}
    return {
        "status": "valid" if bool(valid) else "invalid",
        "reason": sample.get(f"motion_{channel}_reason"),
    }


def _tracking_lost_state(samples: list[Mapping[str, Any]], now_ns: int) -> float | None:
    if not samples or not any("tracking_lost" in sample for sample in samples):
        return None
    ordered = sorted(
        ((timestamp, sample.get("tracking_lost")) for sample in samples if (timestamp := _sample_time(sample)) is not None),
        key=lambda item: item[0],
    )
    started_ns: int | None = None
    last_explicit = False
    for timestamp, value in ordered:
        if not isinstance(value, bool):
            continue
        if value and not last_explicit:
            started_ns = timestamp
        elif not value:
            started_ns = None
        last_explicit = value
    if last_explicit and started_ns is not None:
        return max(0.0, (now_ns - started_ns) / 1_000_000_000.0)
    return 0.0


__all__ = ["WINDOW_NS", "summarize_quality_window"]

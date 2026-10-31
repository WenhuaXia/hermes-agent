"""Scheduler presence: when the cron scheduler process itself was absent per store.

``store_health`` covers the case where the store cannot be WRITTEN (ENOSPC/EROFS/EACCES), but
that state is in-memory: a gateway restart clears it, and a healthy-but-dead scheduler leaves no
trace. So a one-shot whose due instant fell inside the restart gap and is more than
``ONESHOT_GRACE_SECONDS`` late gets retired by the due scan ("removed before firing") instead of
being caught up once.

This module persists a small heartbeat per store directory:

    <cron_dir>/.scheduler_presence.json
        {"last_heartbeat": <epoch|None>, "gap_start": <epoch|None>, "process_start": <epoch>}

``on_startup`` is called by each scheduler process once per store it ticks, before its first
tick: ``gap_start`` becomes the previous process's ``last_heartbeat`` (``None`` on first run),
``process_start`` and ``last_heartbeat`` become now. ``heartbeat`` (called per tick, self-
throttled) refreshes only ``last_heartbeat``. ``gap_covers`` reports whether a due instant fell
inside this process's pre-birth absence window ``[gap_start, process_start]`` (``grace`` slack on
both ends) — the one-shot grace gate consults it so such jobs fire once, late, instead of being
retired.

Presence is deliberately DISK state (unlike ``store_health``): the whole point is surviving a
process death, so the next process reads the previous one's last heartbeat. Every write is
best-effort (a presence failure must never break a tick) and every read fails closed (missing or
corrupt file -> no catch-up, the pre-existing retire behavior).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cron.scheduler")

PRESENCE_FILE = ".scheduler_presence.json"
# Ticks run ~60s apart; never write more often than this (a presence write must stay cheaper and
# less frequent than the jobs.json save it accompanies).
HEARTBEAT_INTERVAL_SECONDS = 60.0

# key (realpath of the cron dir) -> {"record": dict, "last_write": monotonic seconds}
_state: dict[str, dict] = {}
_lock = threading.Lock()

CronDir = os.PathLike | str | None


def _key(cron_dir: CronDir) -> str:
    """Record key for a store: its resolved path (default: the active store)."""
    if cron_dir is None:
        from cron.jobs import _current_cron_store
        cron_dir = _current_cron_store().cron_dir
    return os.path.realpath(str(cron_dir))


def _record_path(key: str) -> Path:
    return Path(key) / PRESENCE_FILE


def _coerce(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load(key: str) -> dict:
    """Tolerant read: missing/corrupt file -> empty record (fails closed, no catch-up)."""
    try:
        raw = json.loads(_record_path(key).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {field: _coerce(raw.get(field)) for field in ("last_heartbeat", "gap_start", "process_start")}


def _persist(key: str, record: dict) -> bool:
    try:
        from utils import atomic_write_text
        if not os.path.isdir(key):
            return False
        atomic_write_text(_record_path(key), json.dumps(record, indent=2) + "\n", tmp_prefix=".presence_")
        return True
    except OSError as exc:
        logger.debug("Scheduler presence write failed for store %s: %s", key, exc)
        return False


def _begin(key: str) -> Optional[dict]:
    """Compute and persist this process's absence gap for the store; None when the write fails.

    ``gap_start`` = the previous process's last heartbeat (None on first run), so the recorded
    gap is exactly the window in which NO scheduler was present before this process was born."""
    now = time.time()
    gap_start = _load(key).get("last_heartbeat")
    record = {"last_heartbeat": now, "gap_start": gap_start, "process_start": now}
    if not _persist(key, record):
        return None
    if gap_start is not None:
        gap = now - gap_start
        if gap >= HEARTBEAT_INTERVAL_SECONDS:
            logger.info(
                "Cron store %s: no scheduler was present for %.0fs before this process started; "
                "one-shots due in that window are caught up once (late), not retired",
                key, gap)
        else:
            # e.g. back-to-back headless `hermes cron tick` runs: normal operation, stay quiet
            logger.debug(
                "Cron store %s: no scheduler was present for %.0fs before this process started",
                key, gap)
    return {"record": record, "last_write": time.monotonic()}


def on_startup(cron_dir: CronDir = None) -> None:
    """Called by each scheduler process once per store it ticks, before its first tick.

    Records the offline gap since the previous process's last heartbeat (``None`` on first run)
    and this process's birth. A process that already recorded this store (this process's earlier
    call, or the tick-path init) is a no-op."""
    try:
        key = _key(cron_dir)
        with _lock:
            if key in _state:
                return
            entry = _begin(key)
            if entry is not None:
                _state[key] = entry
    except Exception:
        logger.debug("Scheduler presence on_startup failed", exc_info=True)


def heartbeat(cron_dir: CronDir = None) -> None:
    """Per-tick presence beat for a store.

    Lazily initializes the store (``on_startup`` semantics) the first time this process observes
    it, then refreshes ``last_heartbeat`` at most once per ``HEARTBEAT_INTERVAL_SECONDS``. Never
    raises: presence is a diagnostic aid, and a failure must leave the tick untouched (the read
    side then fails closed to the pre-existing retire behavior)."""
    try:
        key = _key(cron_dir)
        with _lock:
            entry = _state.get(key)
        if entry is None:
            on_startup(cron_dir)
            return
        with _lock:
            if time.monotonic() - entry["last_write"] < HEARTBEAT_INTERVAL_SECONDS:
                return
            entry["last_write"] = time.monotonic()
            entry["record"]["last_heartbeat"] = time.time()
            record = dict(entry["record"])
        _persist(key, record)
    except Exception:
        logger.debug("Scheduler presence heartbeat failed", exc_info=True)


def gap_covers(cron_dir: CronDir, due_at: float, grace: float) -> bool:
    """Whether a run due at ``due_at`` (epoch seconds) fell inside this store's just-ended
    scheduler-absence window ``[gap_start, process_start]``, allowing ``grace`` slack on both
    ends: a due instant within ``grace`` before the gap (the old scheduler would have fired it on
    its next tick) or within ``grace`` after it (this scheduler's first tick would have).

    Reads the DISK record (not process memory) so the answer survives the restart that created
    the gap; a missing/corrupt record returns False (no catch-up)."""
    try:
        rec = _load(_key(cron_dir))
    except Exception:
        return False
    gap_start = rec.get("gap_start")
    process_start = rec.get("process_start")
    if gap_start is None or process_start is None:
        return False
    return gap_start <= due_at + grace and due_at - grace <= process_start

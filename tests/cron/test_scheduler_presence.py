"""Scheduler presence (cron/scheduler_presence.py): catch-up for one-shots whose due instant
fell inside a window in which the scheduler PROCESS was absent.

``store_health`` covers store-write failures (ENOSPC/EROFS) with in-process memory state — that
state dies with the gateway, so it cannot see the "scheduler process was gone" gap. Presence is
persisted per store (``.scheduler_presence.json``) so the process that survives (or is born) can
compute the offline gap ``[gap_start, process_start]``; the one-shot grace gate then catches up a
job due in that gap once (claim tagged ``late=presence_gap``) instead of retiring it.

Pinned here:
  - on_startup: gap_start = previous process's last_heartbeat (None on first run), process_start = now
  - a second on_startup (simulated restart) derives the gap from the persisted heartbeat
  - gap_covers: due inside the gap -> True (with grace slack on both ends); far before/after ->
    False; first run (gap_start None) / missing / corrupt file -> False
  - a one-shot beyond grace due in the presence gap: caught up, record kept, claim tagged
    late=presence_gap, caught up exactly once
  - beyond grace with no gap, or a STALE gap (no restart: host sleep) -> retired as before
  - heartbeats are throttled; the tick path and the built-in ticker's start() both record presence
"""

import json
import threading
import time
from datetime import datetime, timedelta, UTC

import pytest

from cron import scheduler_presence
from cron.jobs import get_due_jobs, load_jobs, save_jobs

FIXED_NOW = datetime(2026, 6, 22, 12, 0, 0, tzinfo=UTC)
GRACE = 120  # cron.jobs.ONESHOT_GRACE_SECONDS


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir, pin the scan clock, start with no in-process state."""
    monkeypatch.setattr(scheduler_presence, "_state", {})
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _presence_path(cron_store):
    return cron_store / "cron" / scheduler_presence.PRESENCE_FILE


def _write_presence(cron_store, last_heartbeat, gap_start=None, process_start=None):
    (cron_store / "cron").mkdir(parents=True, exist_ok=True)
    _presence_path(cron_store).write_text(json.dumps({
        "last_heartbeat": last_heartbeat,
        "gap_start": gap_start,
        "process_start": process_start,
    }), encoding="utf-8")


def _read_presence(cron_store):
    return json.loads(_presence_path(cron_store).read_text(encoding="utf-8"))


def _oneshot(jid, run_at_dt, *, completed=0, run_claim=None, fire_claim=None):
    return {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "once", "run_at": run_at_dt.isoformat()},
        "next_run_at": run_at_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": 1, "completed": completed},
        "deliver": "local",
        "run_claim": run_claim,
        "fire_claim": fire_claim,
    }


def _patch_wall_and_mono(monkeypatch):
    clock = {"wall": 1_000_000.0, "mono": 0.0}
    monkeypatch.setattr(scheduler_presence.time, "time", lambda: clock["wall"])
    monkeypatch.setattr(scheduler_presence.time, "monotonic", lambda: clock["mono"])
    return clock


class TestOnStartup:
    def test_first_run_records_no_gap(self, cron_store, monkeypatch):
        clock = _patch_wall_and_mono(monkeypatch)
        save_jobs([])
        scheduler_presence.on_startup()
        assert _read_presence(cron_store) == {
            "last_heartbeat": clock["wall"], "gap_start": None, "process_start": clock["wall"],
        }

    def test_restart_derives_gap_from_persisted_heartbeat(self, cron_store, monkeypatch):
        clock = _patch_wall_and_mono(monkeypatch)
        save_jobs([])
        # Process A: startup, then one throttled heartbeat while alive.
        scheduler_presence.on_startup()
        clock["wall"] = 1_000_600.0
        clock["mono"] = 1_000.0
        scheduler_presence.heartbeat()
        assert _read_presence(cron_store)["last_heartbeat"] == 1_000_600.0
        # Process B (restart): fresh in-process state, later wall clock.
        monkeypatch.setattr(scheduler_presence, "_state", {})
        clock["wall"] = 1_000_900.0
        scheduler_presence.on_startup()
        rec = _read_presence(cron_store)
        assert rec["gap_start"] == 1_000_600.0  # A's last heartbeat
        assert rec["process_start"] == 1_000_900.0
        assert rec["last_heartbeat"] == 1_000_900.0

    def test_second_on_startup_in_same_process_is_a_noop(self, cron_store, monkeypatch):
        clock = _patch_wall_and_mono(monkeypatch)
        save_jobs([])
        scheduler_presence.on_startup()
        clock["wall"] = 1_000_500.0
        scheduler_presence.on_startup()
        rec = _read_presence(cron_store)
        assert rec["process_start"] == 1_000_000.0  # first birth, not re-stamped
        assert rec["gap_start"] is None


class TestGapCovers:
    def test_due_inside_gap_covered(self, cron_store):
        _write_presence(cron_store, last_heartbeat=1000, gap_start=900, process_start=1000)
        assert scheduler_presence.gap_covers(cron_store / "cron", 950, GRACE)

    def test_grace_slack_on_both_ends(self, cron_store):
        _write_presence(cron_store, last_heartbeat=1000, gap_start=900, process_start=1000)
        # within GRACE before the gap (the old scheduler would have fired it on its next tick)
        assert scheduler_presence.gap_covers(cron_store / "cron", 900 - GRACE + 10, GRACE)
        # within GRACE after the gap (this scheduler's first tick window)
        assert scheduler_presence.gap_covers(cron_store / "cron", 1000 + GRACE - 10, GRACE)

    def test_due_far_before_or_after_gap_not_covered(self, cron_store):
        _write_presence(cron_store, last_heartbeat=1000, gap_start=900, process_start=1000)
        assert not scheduler_presence.gap_covers(cron_store / "cron", 700, GRACE)    # 700+120 < 900
        assert not scheduler_presence.gap_covers(cron_store / "cron", 1300, GRACE)   # 1300-120 > 1000

    def test_first_run_gap_start_none_not_covered(self, cron_store):
        _write_presence(cron_store, last_heartbeat=1000, gap_start=None, process_start=1000)
        assert not scheduler_presence.gap_covers(cron_store / "cron", 950, GRACE)

    def test_missing_or_corrupt_file_not_covered(self, cron_store):
        assert not scheduler_presence.gap_covers(cron_store / "cron", 950, GRACE)
        (cron_store / "cron").mkdir(parents=True, exist_ok=True)
        _presence_path(cron_store).write_text("not json", encoding="utf-8")
        assert not scheduler_presence.gap_covers(cron_store / "cron", 950, GRACE)


class TestHeartbeatThrottle:
    def test_throttled_and_preserves_gap_fields(self, cron_store, monkeypatch):
        clock = _patch_wall_and_mono(monkeypatch)
        save_jobs([])
        scheduler_presence.on_startup()
        # Inside the throttle window: no write.
        clock["wall"] += 30.0
        clock["mono"] += 30.0
        scheduler_presence.heartbeat()
        assert _read_presence(cron_store)["last_heartbeat"] == 1_000_000.0
        # Past the window: only last_heartbeat advances.
        clock["wall"] += 60.0
        clock["mono"] += 60.0
        scheduler_presence.heartbeat()
        rec = _read_presence(cron_store)
        assert rec["last_heartbeat"] == 1_000_090.0
        assert rec["gap_start"] is None
        assert rec["process_start"] == 1_000_000.0


class TestOneShotPresenceCatchUp:
    def test_due_in_presence_gap_caught_up_once_not_retired(self, cron_store):
        t0 = FIXED_NOW.timestamp()
        # Previous process last heartbeated t0-300; this process was born t0-50. Gap [t0-300,
        # t0-50] covers the due instant t0-200.
        _write_presence(cron_store, last_heartbeat=t0 - 50, gap_start=t0 - 300, process_start=t0 - 50)
        save_jobs([_oneshot("late", FIXED_NOW - timedelta(seconds=200))])

        assert [d["id"] for d in get_due_jobs()] == ["late"]
        stored = {j["id"]: j for j in load_jobs()}
        assert "late" in stored  # NOT retired
        claim = stored["late"]["run_claim"]
        assert claim.get("late") == "presence_gap"
        assert "outage" not in claim
        out_dir = cron_store / "cron" / "output" / "late"
        assert not (out_dir.exists() and any(out_dir.glob("*.md")))  # no "removed before firing"

        # Caught up exactly once: the fresh run_claim makes the next scan skip it.
        assert [d["id"] for d in get_due_jobs()] == []

    def test_no_presence_file_beyond_grace_still_retired(self, cron_store):
        save_jobs([_oneshot("stale", FIXED_NOW - timedelta(hours=3))])
        assert [d["id"] for d in get_due_jobs()] == []
        assert [j["id"] for j in load_jobs()] == []
        out_dir = cron_store / "cron" / "output" / "stale"
        assert list(out_dir.glob("*.md")), "retirement must leave a diagnostic"

    def test_stale_gap_without_restart_still_retired(self, cron_store):
        """Host-sleep regression: the process was NOT restarted (born 2h ago, still running);
        a gap from its birth cannot cover a due instant from the last hour — retire as before."""
        t0 = FIXED_NOW.timestamp()
        _write_presence(cron_store, last_heartbeat=t0 - 3600, gap_start=t0 - 7300, process_start=t0 - 7200)
        save_jobs([_oneshot("sleep", FIXED_NOW - timedelta(seconds=200))])
        assert [d["id"] for d in get_due_jobs()] == []
        assert [j["id"] for j in load_jobs()] == []

    def test_due_well_after_gap_not_covered(self, cron_store):
        """Process born 10 minutes ago: a due instant 150s ago is not in its birth gap — retire."""
        t0 = FIXED_NOW.timestamp()
        _write_presence(cron_store, last_heartbeat=t0 - 600, gap_start=t0 - 700, process_start=t0 - 600)
        save_jobs([_oneshot("after", FIXED_NOW - timedelta(seconds=150))])
        assert [d["id"] for d in get_due_jobs()] == []
        assert [j["id"] for j in load_jobs()] == []


class TestTickWiring:
    def test_tick_records_presence_and_second_tick_is_throttled(self, cron_store, monkeypatch):
        """The tick path (any built-in ticker: gateway, desktop, headless `hermes cron tick`)
        records presence for the store it is admitted on — lazily initializing the gap."""
        from cron import scheduler

        clock = _patch_wall_and_mono(monkeypatch)
        monkeypatch.setattr(scheduler, "_should_yield_tick_to_fresh_gateway", lambda: None)
        monkeypatch.setattr(scheduler, "_sweep_mcp_orphans", lambda: None)
        monkeypatch.setattr(scheduler, "get_due_jobs", list)  # idle tick
        save_jobs([])

        assert scheduler.tick(verbose=False, sync=True) == 0
        first = _read_presence(cron_store)
        assert first == {
            "last_heartbeat": clock["wall"], "gap_start": None, "process_start": clock["wall"],
        }
        # Second tick inside the throttle window: no new write.
        clock["wall"] += 10.0
        clock["mono"] += 10.0
        assert scheduler.tick(verbose=False, sync=True) == 0
        assert _read_presence(cron_store) == first
        # Past the window: only last_heartbeat advances.
        clock["wall"] += 61.0
        clock["mono"] += 61.0
        assert scheduler.tick(verbose=False, sync=True) == 0
        third = _read_presence(cron_store)
        assert third["last_heartbeat"] == clock["wall"]
        assert third["gap_start"] is None and third["process_start"] == first["process_start"]

    def test_reborn_process_gap_file_covers_due_oneshot(self, cron_store):
        """End-to-end at the gate: the disk record shows this process was born after a 200s
        absence (gap [t0-300, t0-100]); a one-shot due at t0-200 is caught up, not retired.
        No in-process state involved: gap_covers reads the disk, which is what survives the
        restart that created the gap."""
        t0 = FIXED_NOW.timestamp()
        _write_presence(cron_store, last_heartbeat=t0 - 100, gap_start=t0 - 300, process_start=t0 - 100)
        save_jobs([_oneshot("late", FIXED_NOW - timedelta(seconds=200))])
        assert [d["id"] for d in get_due_jobs()] == ["late"]
        stored = {j["id"]: j for j in load_jobs()}
        assert stored["late"]["run_claim"].get("late") == "presence_gap"


class TestProviderStartupWiring:
    def _run_one_tick(self, monkeypatch, tick_fn):
        """Run one iteration of the built-in ticker loop with a stubbed tick (the startup block
        — recover_interrupted + heartbeat + presence on_startup — runs before it)."""
        from cron.scheduler_provider import InProcessCronScheduler

        provider = InProcessCronScheduler()
        monkeypatch.setattr(provider, "recover_interrupted", lambda: 0)
        monkeypatch.setattr("cron.scheduler.tick", tick_fn)
        stop_event = threading.Event()
        original_wait = stop_event.wait

        def _stop_after_first_tick(timeout=None):
            stop_event.set()
            return original_wait(0)

        stop_event.wait = _stop_after_first_tick
        provider.start(stop_event)

    def test_startup_records_presence(self, cron_store, monkeypatch):
        save_jobs([])
        self._run_one_tick(monkeypatch, lambda **kw: 0)
        rec = _read_presence(cron_store)
        assert rec["gap_start"] is None
        assert rec["process_start"] is not None
        assert rec["last_heartbeat"] is not None

    def test_restart_startup_records_gap_since_previous_heartbeat(self, cron_store, monkeypatch):
        now = time.time()
        _write_presence(cron_store, last_heartbeat=now - 300, gap_start=None, process_start=now - 3600)
        save_jobs([])
        self._run_one_tick(monkeypatch, lambda **kw: 0)
        rec = _read_presence(cron_store)
        assert rec["gap_start"] == now - 300  # previous process's last heartbeat
        assert rec["process_start"] >= now - 5
        assert rec["last_heartbeat"] == rec["process_start"]

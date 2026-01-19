#!/usr/bin/env python3
"""
comp_time.py

Find runs under ../trained matching a pattern (default: 2b-lr0.01_p16_*)
Parse each run's train.log and compute wall-clock computation time per epoch.
Then report per-run times and global min/max/avg stats.

Usage:
  python comp_time.py
  python comp_time.py --pattern "2b-lr0.01_p16_*"
  python comp_time.py --trained_dir ../trained --pattern "2b-lr0.01_p16_*"

Notes:
- Epoch duration is measured from:
    start_time = timestamp at "***** Running training *****"
    end_time   = timestamp at "val epoch <k>:"
  So epoch_k_time = end_time(k) - end_time(k-1), with end_time(0)=start_time.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean


TS_RE = re.compile(r"^(?P<ts>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - root -\s+(?P<msg>.*)$")
START_RE = re.compile(r"\*{5} Running training \*{5}")
VAL_EPOCH_RE = re.compile(r"val epoch (?P<epoch>\d+):")


@dataclass
class RunTiming:
    run_name: str
    log_path: Path
    epoch_seconds: list[float]  # per-epoch wall time in seconds

    @property
    def num_epochs(self) -> int:
        return len(self.epoch_seconds)

    @property
    def mean_epoch_seconds(self) -> float:
        return mean(self.epoch_seconds) if self.epoch_seconds else float("nan")


def parse_train_log(log_path: Path) -> tuple[datetime | None, dict[int, datetime]]:
    """
    Returns:
      start_ts: timestamp for "***** Running training *****" (or None)
      val_ts: dict epoch -> timestamp for "val epoch k:"
    """
    start_ts: datetime | None = None
    val_ts: dict[int, datetime] = {}

    # Read as text; ignore decoding issues
    for line in log_path.read_text(errors="ignore").splitlines():
        m = TS_RE.match(line)
        if not m:
            continue

        ts = datetime.strptime(m.group("ts"), "%m/%d/%Y %H:%M:%S")
        msg = m.group("msg")

        if start_ts is None and START_RE.search(msg):
            start_ts = ts
            continue

        vm = VAL_EPOCH_RE.search(msg)
        if vm:
            epoch = int(vm.group("epoch"))
            # Keep the first occurrence per epoch (should be unique anyway)
            val_ts.setdefault(epoch, ts)

    return start_ts, val_ts


def seconds_to_hhmmss(sec: float) -> str:
    if sec != sec:  # NaN
        return "NaN"
    td = timedelta(seconds=int(round(sec)))
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def compute_epoch_times(start_ts: datetime | None, val_ts: dict[int, datetime]) -> list[float]:
    """
    Compute epoch durations using:
      epoch1 = val(1) - start
      epochk = val(k) - val(k-1)
    Only computes contiguous epochs starting at 1.
    """
    if start_ts is None:
        return []

    # Determine how many epochs we can compute contiguously
    k = 1
    epoch_times: list[float] = []
    prev = start_ts

    while k in val_ts:
        cur = val_ts[k]
        epoch_times.append((cur - prev).total_seconds())
        prev = cur
        k += 1

    return epoch_times


def find_runs(trained_dir: Path, pattern: str) -> list[Path]:
    # Only directories directly under trained_dir matching pattern
    return sorted([p for p in trained_dir.glob(pattern) if p.is_dir()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained_dir", type=Path, default=Path("../trained"), help="Path to trained directory (default: ../trained)")
    ap.add_argument("--pattern", type=str, default="2b-lr0.01_p16_*", help='Run folder glob pattern (default: "2b-lr0.01_p16_*")')
    ap.add_argument("--log_name", type=str, default="train.log", help='Log file name inside each run folder (default: "train.log")')
    args = ap.parse_args()

    trained_dir: Path = args.trained_dir
    pattern: str = args.pattern
    log_name: str = args.log_name

    runs = find_runs(trained_dir, pattern)
    if not runs:
        print(f"[ERROR] No runs found: {trained_dir.resolve()}/{pattern}")
        return

    timings: list[RunTiming] = []
    skipped: list[tuple[str, str]] = []  # (run, reason)

    for run_dir in runs:
        log_path = run_dir / log_name
        if not log_path.exists():
            skipped.append((run_dir.name, f"missing {log_name}"))
            continue

        start_ts, val_ts = parse_train_log(log_path)
        epoch_times = compute_epoch_times(start_ts, val_ts)

        if not epoch_times:
            reason = "could not parse start/val epoch timestamps"
            if start_ts is None:
                reason = "missing '***** Running training *****' timestamp"
            elif 1 not in val_ts:
                reason = "missing 'val epoch 1' timestamp"
            skipped.append((run_dir.name, reason))
            continue

        timings.append(RunTiming(run_dir.name, log_path, epoch_times))

    if skipped:
        print("===== SKIPPED RUNS =====")
        for name, reason in skipped:
            print(f"- {name}: {reason}")
        print()

    if not timings:
        print("[ERROR] No runnable logs parsed successfully.")
        return

    # Aggregate stats
    all_epoch_seconds = [s for t in timings for s in t.epoch_seconds]
    per_run_mean_seconds = [t.mean_epoch_seconds for t in timings]

    def stats(xs: list[float]) -> tuple[float, float, float]:
        return (min(xs), mean(xs), max(xs))

    all_min, all_avg, all_max = stats(all_epoch_seconds)
    run_min, run_avg, run_max = stats(per_run_mean_seconds)

    print("===== SUMMARY (across all parsed runs) =====")
    print(f"Runs parsed: {len(timings)} / {len(runs)}")
    print("")
    print("Per-epoch time (ALL individual epochs across all runs):")
    print(f"  min: {seconds_to_hhmmss(all_min)}")
    print(f"  avg: {seconds_to_hhmmss(all_avg)}")
    print(f"  max: {seconds_to_hhmmss(all_max)}")



if __name__ == "__main__":
    main()

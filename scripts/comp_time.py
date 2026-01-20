#!/usr/bin/env python3
"""
comp_time_compare.py  (you can name this comp_time.py if you want)

Compare wall-clock epoch time between prefix vs LoRA runs using box plots.

Epoch duration definition:
  epoch1 = ts("val epoch 1") - ts("***** Running training *****")
  epochk = ts("val epoch k") - ts("val epoch k-1")

Method detection:
  parsed from train.log line:
    "Training args Namespace(... model_type='prefix' ...)"  or  model_type='lora'

Also extracts (for x-axis label):
  "Fraction of trainable parameters = <float>"
and x-axis labels become:
  prefix (<avg_frac>%)
  lora   (<avg_frac>%)

Output filename auto-includes model tag inferred from --patterns:
  default out:
    comp_time_boxplot_<modeltag>.png
  examples:
    comp_time_boxplot_350m.png
    comp_time_boxplot_2b.png

Usage examples (run from scripts/):
  python comp_time.py
  python comp_time.py --patterns "2b-lr0.01_p16_*" "2b-lr0.0001_r8_*"
  python comp_time.py --patterns "350m-lr*_p*" "350m-lr*_r*" --use per_run_mean
  python comp_time.py --patterns "350m-*" --out my_custom.png   # explicit override
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Literal

import matplotlib.pyplot as plt


TS_RE = re.compile(r"^(?P<ts>\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}) - INFO - root -\s+(?P<msg>.*)$")
START_RE = re.compile(r"\*{5} Running training \*{5}")
VAL_EPOCH_RE = re.compile(r"val epoch (?P<epoch>\d+):")
MODEL_TYPE_RE = re.compile(r"Training args Namespace\(.+?model_type='(?P<mt>[^']+)'")
FRACTION_TRAINABLE_RE = re.compile(r"Fraction of trainable parameters\s*=\s*(?P<f>[0-9]*\.?[0-9]+)")


@dataclass
class RunTiming:
    run_name: str
    method: str                    # 'prefix' or 'lora'
    log_path: Path
    epoch_seconds: list[float]     # per-epoch wall time in seconds
    frac_trainable_percent: float  # already percent units, e.g. 0.1886 means 0.1886%

    @property
    def num_epochs(self) -> int:
        return len(self.epoch_seconds)

    @property
    def mean_epoch_seconds(self) -> float:
        return mean(self.epoch_seconds) if self.epoch_seconds else float("nan")


def infer_model_tag(patterns: list[str]) -> str:
    """
    Infer model size tag (e.g., 350m, 2b, 6b) from patterns.
    Returns 'unknown' if nothing is detected.
    """
    joined = " ".join(patterns).lower()
    for tag in ["350m", "2b", "6b", "7b", "13b"]:
        if tag in joined:
            return tag
    return "unknown"


def seconds_to_hhmmss(sec: float) -> str:
    if sec != sec:  # NaN
        return "NaN"
    td = timedelta(seconds=int(round(sec)))
    total = int(td.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_train_log(log_path: Path) -> tuple[str | None, datetime | None, dict[int, datetime], float | None]:
    """
    Returns:
      model_type: 'prefix'/'lora'/... or None
      start_ts: timestamp for "***** Running training *****" (or None)
      val_ts: dict epoch -> timestamp for "val epoch k:"
      frac_trainable_percent: float (already in %) from log (e.g. 0.1886), or None
    """
    model_type: str | None = None
    start_ts: datetime | None = None
    val_ts: dict[int, datetime] = {}
    frac_trainable: float | None = None

    for line in log_path.read_text(errors="ignore").splitlines():
        # model_type can appear on a non-timestamp line
        if model_type is None:
            mm = MODEL_TYPE_RE.search(line)
            if mm:
                model_type = mm.group("mt").strip()

        # fraction can be extracted from any line (timestamped or not)
        if frac_trainable is None:
            fm = FRACTION_TRAINABLE_RE.search(line)
            if fm:
                frac_trainable = float(fm.group("f"))

        # Parse timestamps for epoch timing
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
            val_ts.setdefault(epoch, ts)

    return model_type, start_ts, val_ts, frac_trainable


def compute_epoch_times(start_ts: datetime | None, val_ts: dict[int, datetime]) -> list[float]:
    """
    epoch1 = val(1) - start
    epochk = val(k) - val(k-1)
    Only contiguous epochs starting at 1.
    """
    if start_ts is None:
        return []

    k = 1
    epoch_times: list[float] = []
    prev = start_ts

    while k in val_ts:
        cur = val_ts[k]
        epoch_times.append((cur - prev).total_seconds())
        prev = cur
        k += 1

    return epoch_times


def find_runs(trained_dir: Path, patterns: list[str]) -> list[Path]:
    # Merge globs; unique; only dirs directly under trained_dir
    out: dict[str, Path] = {}
    for pat in patterns:
        for p in trained_dir.glob(pat):
            if p.is_dir():
                out[p.name] = p
    return [out[k] for k in sorted(out.keys())]


def summarize(label: str, xs: list[float]) -> str:
    if not xs:
        return f"{label}: (no data)"
    return (
        f"{label}: "
        f"n={len(xs)}, "
        f"min={seconds_to_hhmmss(min(xs))}, "
        f"avg={seconds_to_hhmmss(mean(xs))}, "
        f"max={seconds_to_hhmmss(max(xs))}"
    )


def fmt_frac_label(fracs: list[float], digits: int = 4) -> str:
    """
    fracs are already in percent units, e.g. 0.1886 means 0.1886%.
    We show average, and if there is variation, show min-max too.
    """
    if not fracs:
        return "N/A"
    avg = mean(fracs)
    mn = min(fracs)
    mx = max(fracs)
    if abs(mx - mn) < 1e-9:
        return f"{avg:.{digits}f}%"
    return f"{avg:.{digits}f}% [{mn:.{digits}f}–{mx:.{digits}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained_dir", type=Path, default=Path("../trained"))
    ap.add_argument("--patterns", nargs="+", default=["2b-*"], help='One or more run folder glob patterns (default: "2b-*")')
    ap.add_argument("--log_name", type=str, default="train.log")
    ap.add_argument("--out", type=Path, default=Path("comp_time_boxplot.png"), help="Output PNG path")
    ap.add_argument(
        "--use",
        choices=["per_epoch", "per_run_mean"],
        default="per_epoch",
        help="Boxplot data: all epochs (per_epoch) or per-run mean epoch time (per_run_mean)",
    )
    ap.add_argument("--title", type=str, default="Epoch time comparison")
    ap.add_argument("--frac_digits", type=int, default=4, help="Digits for trainable fraction (%) in x-labels")
    args = ap.parse_args()

    trained_dir: Path = args.trained_dir
    patterns: list[str] = args.patterns
    log_name: str = args.log_name
    use: Literal["per_epoch", "per_run_mean"] = args.use

    model_tag = infer_model_tag(patterns)

    # Auto-name output if user didn't override --out
    if args.out == Path("comp_time_boxplot.png"):
        out_path = Path(f"comp_time_boxplot_{model_tag}.png")
    else:
        out_path = args.out

    runs = find_runs(trained_dir, patterns)
    if not runs:
        print(f"[ERROR] No runs found in {trained_dir.resolve()} with patterns={patterns}")
        return

    timings: list[RunTiming] = []
    skipped: list[tuple[str, str]] = []

    for run_dir in runs:
        log_path = run_dir / log_name
        if not log_path.exists():
            skipped.append((run_dir.name, f"missing {log_name}"))
            continue

        model_type, start_ts, val_ts, frac_trainable = parse_train_log(log_path)
        if model_type is None:
            skipped.append((run_dir.name, "could not find model_type in log"))
            continue

        epoch_times = compute_epoch_times(start_ts, val_ts)
        if not epoch_times:
            reason = "could not parse start/val epoch timestamps"
            if start_ts is None:
                reason = "missing '***** Running training *****' timestamp"
            elif 1 not in val_ts:
                reason = "missing 'val epoch 1' timestamp"
            skipped.append((run_dir.name, reason))
            continue

        if frac_trainable is None:
            skipped.append((run_dir.name, "missing 'Fraction of trainable parameters' in log"))
            continue

        timings.append(RunTiming(run_dir.name, model_type, log_path, epoch_times, frac_trainable))

    if skipped:
        print("===== SKIPPED RUNS =====")
        for name, reason in skipped:
            print(f"- {name}: {reason}")
        print()

    if not timings:
        print("[ERROR] No runnable logs parsed successfully.")
        return

    # Group by method
    by_method: dict[str, list[RunTiming]] = {}
    for t in timings:
        by_method.setdefault(t.method, []).append(t)

    methods_sorted = sorted(by_method.keys(), key=lambda x: {"prefix": 0, "lora": 1}.get(x, 99))

    # Build data vectors + fraction vectors per method
    data: list[list[float]] = []
    method_fracs: list[list[float]] = []

    for m in methods_sorted:
        runs_m = by_method[m]
        fracs_m = [t.frac_trainable_percent for t in runs_m]
        method_fracs.append(fracs_m)

        if use == "per_epoch":
            xs = [s for t in runs_m for s in t.epoch_seconds]
        else:
            xs = [t.mean_epoch_seconds for t in runs_m]
        data.append(xs)

    # Print summaries (now includes trainable fraction label)
    print("===== SUMMARY BY METHOD =====")
    for m, xs, frs in zip(methods_sorted, data, method_fracs):
        print(summarize(m, xs) + f" | trainable_frac={fmt_frac_label(frs, digits=args.frac_digits)}")
    print()

    # X-axis labels: method + trainable fraction
    labels = [
        f"{m} ({fmt_frac_label(frs, digits=args.frac_digits)})"
        for m, frs in zip(methods_sorted, method_fracs)
    ]

    # Box plot
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Epoch time (seconds)" if use == "per_epoch" else "Mean epoch time per run (seconds)")
    plt.title(f"{args.title} ({model_tag}) [{use}]")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    print(f"[OK] Saved boxplot: {out_path.resolve()}")


if __name__ == "__main__":
    main()

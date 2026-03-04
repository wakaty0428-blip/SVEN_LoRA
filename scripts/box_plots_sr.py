#!/usr/bin/env python3
"""
box_plots_sr.py

Make 3 box plots of OVERALL SECURITY RATE (SR):
  1) Original  (control=orig)  from original LM txts
  2) Prefix    (control=sec/vul) from prefix txts
  3) LoRA      (control=sec/vul) from lora txts

X-axis labels are exactly:
  "Original", "Prefix", "LoRA"

Security-rate txt situation:
- Original LM: files like 350m-lm-1.txt ... 350m-lm-10.txt contain "orig" control:
    | overall | overall | orig | 58.6, ... |
- Prefix: one txt file contains BOTH sec + vul rows
    e.g., 350m-ep7-lr0.01_p8_lm0.180_con41_kl410-1.txt
- LoRA: files are split by control
    e.g., 350m-ep7-...-sec-1.txt, 350m-ep7-...-vul-1.txt

CLI:
- --control : sec | vul
- --scripts_dir : directory containing the SR .txt files
- --out : output PNG filename (saved under scripts_dir/box_plots/)

Example (run from scripts/):
  python box_plots_sr.py --scripts_dir 350m --control sec --out sr_sec_original_prefix_lora.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


# =========================================================
# EDIT THESE PATTERNS FOR YOUR TARGET MODEL/SWEEP
# (All patterns are relative to --scripts_dir)
# =========================================================
PATTERNS = {
    # NEW: original LM patterns (10 trials: 1..10)
    # Change to "350m-lm-*.txt" if your LM files are named that way.
    "original": {
        "orig": ["350m-lm-*.txt"],
    },
    "prefix": {
        "any": ["350m-ep7-lr0.01_p16_lm0.180_con41_kl410-*.txt"],
    },
    "lora": {
        "sec": ["350m-ep7-lr1e-04_r8_a16_ld0.1_tqkv_lm0.180_con41_kl410-sec-*.txt"],
        "vul": ["350m-ep7-lr1e-04_r8_a16_ld0.1_tqkv_lm0.180_con41_kl410-vul-*.txt"],
    },
}


def find_txt_files(base: Path, method: str, control: str) -> List[Path]:
    """
    Returns list of matched txt files for:
      - method="original"  (control ignored; uses PATTERNS["original"]["orig"])
      - method="prefix"    (control ignored; uses PATTERNS["prefix"]["any"])
      - method="lora"      (control used; PATTERNS["lora"][control])
    """
    files: List[Path] = []

    if method == "original":
        for pat in PATTERNS["original"]["orig"]:
            files.extend(base.glob(pat))

    elif method == "prefix":
        for pat in PATTERNS["prefix"]["any"]:
            files.extend(base.glob(pat))

    elif method == "lora":
        for pat in PATTERNS["lora"][control]:
            files.extend(base.glob(pat))

    else:
        raise ValueError(f"Unknown method: {method}")

    return sorted(set(files))


def parse_overall_rate(txt_path: Path, control: str) -> Optional[float]:
    """
    Parse overall security rate for the specified control ("sec", "vul", or "orig") from a single txt.
    Expected row:
      | overall | overall | sec  | 84.0, ... |
      | overall | overall | vul  | 35.3, ... |
      | overall | overall | orig | 58.6, ... |
    """
    want = control.lower()
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("| overall"):
            continue

        cols = [c.strip() for c in line.split("|") if c.strip()]
        if len(cols) < 4:
            continue

        if cols[2].lower() != want:
            continue

        head = cols[3].split(",", 1)[0].strip()
        m = re.search(r"[-+]?\d+(?:\.\d+)?", head)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None

    return None


def prepare_output_path(scripts_dir: Path, out_name: str) -> Path:
    out_dir = scripts_dir / "box_plots"
    out_dir.mkdir(exist_ok=True)
    return out_dir / out_name


def summarize(values: List[float]) -> str:
    if not values:
        return "(no values)"
    mean = sum(values) / len(values)
    return f"n={len(values)} | min={min(values):.2f} mean={mean:.2f} max={max(values):.2f}"


def collect_values(scripts_dir: Path, method: str, control: str, label: str) -> List[float]:
    files = find_txt_files(scripts_dir, method, control)
    values: List[float] = []

    # original always parsed with "orig"
    parse_control = "orig" if method == "original" else control

    for f in files:
        v = parse_overall_rate(f, parse_control)
        if v is not None:
            values.append(v)

    print(f"\n=== {label} ({parse_control}) ===")
    print(f"files checked : {len(files)}")
    print(f"sr values     : {summarize(values)}")
    return values


def _auto_ylim(vals: List[float], pad: float = 0.8) -> Tuple[float, float]:
    if not vals:
        return (0.0, 1.0)
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        return (lo - 1.0, hi + 1.0)
    return (lo - pad, hi + pad)


def make_boxplot(
    data: List[List[float]],
    labels: List[str],
    out_path: Path,
    title: str,
) -> None:
    all_vals = [v for group in data for v in group]
    if not all_vals:
        print("[ERROR] No data to plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.boxplot(data, labels=labels, showmeans=True)

    ylo, yhi = _auto_ylim(all_vals, pad=0.8)
    ax.set_ylim(ylo, yhi)

    ax.set_ylabel("Overall security rate")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--control",
        choices=["sec", "vul"],
        default="sec",
        help="Which control(s) to parse for Prefix/LoRA. (Original always uses orig.)",
    )
    ap.add_argument(
        "--scripts_dir",
        default=".",
        help="Directory containing the SR .txt files (e.g., ./scripts/350m or ./scripts/350m).",
    )
    ap.add_argument(
        "--out",
        default="boxplot_sr.png",
        help="Output PNG filename (saved under scripts_dir/box_plots/).",
    )
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    c = args.control

    # Collect in your required order: Original, Prefix, LoRA
    original_vals = collect_values(scripts_dir, "original", c, "Original")
    prefix_vals = collect_values(scripts_dir, "prefix", c, "Prefix")
    lora_vals = collect_values(scripts_dir, "lora", c, "LoRA")

    plot_data = [original_vals, prefix_vals, lora_vals]
    plot_labels = ["Original", "Prefix", "LoRA"]

    # Require all 3 to have values to make a clean 3-box plot
    if any(len(d) == 0 for d in plot_data):
        print("\n[ERROR] At least one of (Original/Prefix/LoRA) has no parsed values.")
        print("        Fix PATTERNS so all three match files in --scripts_dir.")
        print(f"        scripts_dir = {scripts_dir}")
        print(f"        control     = {c} (Original uses orig regardless)")
        return

    title = f"Overall Security Rate Distribution ({c})"
    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path, title=title)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
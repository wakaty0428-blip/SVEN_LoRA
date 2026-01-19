#!/usr/bin/env python3
"""
box_plots_sr.py

Make box plots of OVERALL SECURITY RATE (SR).

Security-rate txt situation:
- Prefix: one txt file contains BOTH sec + vul rows
    e.g., 350m-lr0.01_p16_lm0.180_con41_kl410.txt
- LoRA: files are split by control
    e.g., 350m-lr0.0001_r8_lm0.180_con41_kl410-sec.txt
          350m-lr0.0001_r8_lm0.180_con41_kl410-vul.txt

CLI:
- --method  : prefix | lora | both
- --control : sec | vul | both

Plot layout:
- If --control sec:
    method=both   -> [prefix/sec, lora/sec]
    method=prefix -> [prefix/sec]
    method=lora   -> [lora/sec]

- If --control vul:
    method=both   -> [prefix/vul, lora/vul]
    method=prefix -> [prefix/vul]
    method=lora   -> [lora/vul]

- If --control both:
    method=both   -> [prefix/sec, lora/sec, prefix/vul, lora/vul]  (sec left, vul right)
    method=prefix -> [prefix/sec, prefix/vul]
    method=lora   -> [lora/sec, lora/vul]

Outputs are saved under:
  <scripts_dir>/box_plots/<out>

Examples (run from scripts/):
  python box_plots_sr.py --method both --control sec  --out sr_sec_prefix_vs_lora.png
  python box_plots_sr.py --method both --control vul  --out sr_vul_prefix_vs_lora.png
  python box_plots_sr.py --method both --control both --out sr_sec_vul_grouped.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


# -----------------------
# File name patterns (based on your examples)
# -----------------------
PATTERNS = {
    "prefix": {
        # Example: 350m-lr0.01_p16_lm0.180_con41_kl410.txt
        "any": ["350m-lr*_p*_lm*_con*_kl*.txt"],
    },
    "lora": {
        # Example: 350m-lr0.0001_r8_lm0.180_con41_kl410-sec.txt
        "sec": ["350m-lr*_r*_lm*_con*_kl*-sec.txt"],
        "vul": ["350m-lr*_r*_lm*_con*_kl*-vul.txt"],
    },
}


def find_txt_files(base: Path, method: str, control: str) -> List[Path]:
    """
    For prefix: one file contains both sec+vul -> use 'any' pattern regardless of control.
    For lora: control-specific suffix.
    """
    files: List[Path] = []
    if method == "prefix":
        for pat in PATTERNS["prefix"]["any"]:
            files.extend(base.glob(pat))
    else:
        for pat in PATTERNS["lora"][control]:
            files.extend(base.glob(pat))
    return sorted(set(files))


def parse_overall_rate(txt_path: Path, control: str) -> Optional[float]:
    """
    Parse overall security rate for the specified control ("sec" or "vul") from a single txt.

    Expected row:
      | overall | overall | sec | 84.0, 0.0, 100.0 | ...
      | overall | overall | vul | 35.3, 0.0, 100.0 | ...
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

        sec_rate_cell = cols[3]  # e.g. "84.0,   0.0,     100.0"
        head = sec_rate_cell.split(",", 1)[0].strip()
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


def make_boxplot(data: List[List[float]], labels: List[str], out_path: Path, title: str) -> None:
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Overall security rate")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize(values: List[float]) -> str:
    if not values:
        return "(no values)"
    mean = sum(values) / len(values)
    return f"n={len(values)} | min={min(values):.2f} mean={mean:.2f} max={max(values):.2f}"


def collect_values(scripts_dir: Path, method: str, control: str) -> List[float]:
    files = find_txt_files(scripts_dir, method, control)
    values: List[float] = []
    for f in files:
        v = parse_overall_rate(f, control)
        if v is not None:
            values.append(v)

    print(f"\n=== {method.upper()}/{control} ===")
    print(f"files checked : {len(files)}")
    print(f"sr values     : {summarize(values)}")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["prefix", "lora", "both"], default="both",
                    help="Which method(s) to include.")
    ap.add_argument("--control", choices=["sec", "vul", "both"], default="sec",
                    help="Which control(s) to include.")
    ap.add_argument("--scripts_dir", default=".",
                    help="Directory containing the SR .txt files (e.g., ./scripts).")
    ap.add_argument("--out", default="boxplot_sr.png",
                    help="Output PNG filename (saved under scripts_dir/box_plots/).")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    methods = ["prefix", "lora"] if args.method == "both" else [args.method]

    plot_data: List[List[float]] = []
    plot_labels: List[str] = []

    if args.control == "both":
        # sec group (left), then vul group (right)
        for c in ["sec", "vul"]:
            for m in methods:
                plot_data.append(collect_values(scripts_dir, m, c))
                plot_labels.append(f"{m}/{c}")
        title = "Overall Security Rate Distribution (sec left, vul right)"
    else:
        c = args.control
        for m in methods:
            plot_data.append(collect_values(scripts_dir, m, c))
            plot_labels.append(f"{m}/{c}")
        title = f"Overall Security Rate Distribution"

    if all(len(d) == 0 for d in plot_data):
        print("\n[ERROR] No security rates parsed. Nothing to plot.")
        return

    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path, title=title)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()

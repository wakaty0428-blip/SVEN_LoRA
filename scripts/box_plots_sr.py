#!/usr/bin/env python3
"""
box_plots.py

Make box plots of OVERALL SECURITY RATE (SEC ONLY).

- Prefix: one txt file contains sec + vul → we ONLY take sec
- LoRA: sec/vul may be separated → we ONLY take sec (we match *-sec.txt)

Extracts from rows like:
| overall | overall | sec | 84.0, 0.0, 100.0 | ...

Outputs are saved under:
  <scripts_dir>/box_plots/<out>

Examples (run from scripts/):
  python box_plots.py --method both   --out prefix_vs_lora.png
  python box_plots.py --method prefix --out prefix.png
  python box_plots.py --method lora   --out lora.png
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
    # Example: 350m-lr0.01_p16_lm0.180_con41_kl410.txt
    "prefix": ["350m-lr*_p*_lm*_con*_kl*.txt"],
    # Example: 350m-lr0.0001_r8_lm0.180_con41_kl410-sec.txt
    "lora": ["350m-lr*_r*_lm*_con*_kl*-sec.txt"],
}


def find_txt_files(base: Path, method: str) -> List[Path]:
    files: List[Path] = []
    for pat in PATTERNS[method]:
        files.extend(base.glob(pat))
    return sorted(set(files))


def parse_overall_sec_rate(txt_path: Path) -> Optional[float]:
    """
    Return overall security rate (control=sec) from a single txt file.
    Only parses the row starting with "| overall" and control "sec".
    """
    text = txt_path.read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        line = line.strip()

        # Only care about overall rows
        if not line.startswith("| overall"):
            continue

        # Split markdown-table row into columns
        cols = [c.strip() for c in line.split("|") if c.strip()]
        # cols expected: [cwe, scenario, control, sec_rate, sec_mean, total_mean, dup_mean, non_parsed_mean]
        if len(cols) < 4:
            continue

        control = cols[2].lower()
        if control != "sec":
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
    """
    Create <scripts_dir>/box_plots if needed, and return full output path inside it.
    """
    out_dir = scripts_dir / "box_plots"
    out_dir.mkdir(exist_ok=True)
    return out_dir / out_name


def make_boxplot(data: List[List[float]], labels: List[str], out_path: Path) -> None:
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Overall security rate")
    plt.title("Overall Security Rate Distribution")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["prefix", "lora", "both"], default="both",
                    help="Which method(s) to plot.")
    ap.add_argument("--scripts_dir", default=".",
                    help="Directory containing the .txt files (e.g., ./scripts).")
    ap.add_argument("--out", default="boxplot_security_rate.png",
                    help="Output PNG filename (saved under scripts_dir/box_plots/).")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    methods = ["prefix", "lora"] if args.method == "both" else [args.method]

    plot_data: List[List[float]] = []
    plot_labels: List[str] = []

    for m in methods:
        files = find_txt_files(scripts_dir, m)
        values: List[float] = []

        for f in files:
            v = parse_overall_sec_rate(f)
            if v is not None:
                values.append(v)

        # Console summary
        print(f"\n=== {m.upper()} ===")
        print(f"files checked : {len(files)}")
        print(f"sec values    : {len(values)}")
        if values:
            mean = sum(values) / len(values)
            print(f"min={min(values):.2f}, mean={mean:.2f}, max={max(values):.2f}")
        else:
            print("[WARN] No values parsed for this method. Check patterns or file contents.")

        plot_data.append(values)
        plot_labels.append(m)

    if all(len(d) == 0 for d in plot_data):
        print("\n[ERROR] No overall/sec security rates parsed. Nothing to plot.")
        return

    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()

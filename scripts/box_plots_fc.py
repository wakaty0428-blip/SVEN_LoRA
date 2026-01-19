#!/usr/bin/env python3
"""
box_plots_fc.py

Make box plots of FUNCTIONAL CORRECTNESS (HumanEval) using pass@100.

HumanEval txt files are split by control:
  human-eval-...-sec.txt
  human-eval-...-vul.txt

CLI:
- --method  : prefix | lora | both
- --control : sec | vul | both

Plot layout:
- If --control sec:
    - method=both  -> [prefix/sec, lora/sec]
    - method=prefix-> [prefix/sec]
    - method=lora  -> [lora/sec]

- If --control vul:
    - method=both  -> [prefix/vul, lora/vul]
    - method=prefix-> [prefix/vul]
    - method=lora  -> [lora/vul]

- If --control both:
    - method=both  -> [prefix/sec, lora/sec, prefix/vul, lora/vul] (sec left, vul right)
    - method=prefix-> [prefix/sec, prefix/vul]
    - method=lora  -> [lora/sec, lora/vul]

Outputs are saved under:
  <scripts_dir>/box_plots/<out>
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


PATTERNS = {
    "prefix": {
        "sec": ["human-eval-350m-lr*_p*_lm*_con*_kl*-sec.txt"],
        "vul": ["human-eval-350m-lr*_p*_lm*_con*_kl*-vul.txt"],
    },
    "lora": {
        "sec": ["human-eval-350m-lr*_r*_lm*_con*_kl*-sec.txt"],
        "vul": ["human-eval-350m-lr*_r*_lm*_con*_kl*-vul.txt"],
    },
}


def find_txt_files(base: Path, method: str, control: str) -> List[Path]:
    files: List[Path] = []
    for pat in PATTERNS[method][control]:
        files.extend(base.glob(pat))
    return sorted(set(files))


def parse_pass_at_100(txt_path: Path) -> Optional[float]:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    header_idx = None
    pass100_col = None

    for i, ln in enumerate(lines):
        if ln.startswith("|") and "pass@100" in ln:
            cols = [c.strip() for c in ln.split("|") if c.strip()]
            for j, c in enumerate(cols):
                if c.replace(" ", "").lower() == "pass@100":
                    pass100_col = j
                    header_idx = i
                    break
        if header_idx is not None:
            break

    if header_idx is None or pass100_col is None:
        return None

    sep_re = re.compile(r"^\|\s*[-:+]+\s*(\|\s*[-:+]+\s*)+\|?$")
    for ln in lines[header_idx + 1 :]:
        if sep_re.match(ln):
            continue
        if not ln.startswith("|"):
            continue

        cols = [c.strip() for c in ln.split("|") if c.strip()]
        if pass100_col >= len(cols):
            continue

        cell = cols[pass100_col]
        m = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
        if not m:
            continue
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
    plt.ylabel("Functional correctness (pass@100)")
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
        v = parse_pass_at_100(f)
        if v is not None:
            values.append(v)

    print(f"\n=== {method.upper()}/{control} ===")
    print(f"files checked : {len(files)}")
    print(f"pass@100 vals : {summarize(values)}")
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["prefix", "lora", "both"], default="both",
                    help="Which method(s) to include.")
    ap.add_argument("--control", choices=["sec", "vul", "both"], default="both",
                    help="Which control(s) to include.")
    ap.add_argument("--scripts_dir", default=".",
                    help="Directory containing the human-eval txt files (e.g., ./scripts).")
    ap.add_argument("--out", default="boxplot_fc.png",
                    help="Output PNG filename (saved under scripts_dir/box_plots/).")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()

    methods = ["prefix", "lora"] if args.method == "both" else [args.method]
    controls = ["sec", "vul"] if args.control == "both" else [args.control]

    plot_data: List[List[float]] = []
    plot_labels: List[str] = []

    # Ordering rules:
    # - If control=both and method=both: prefix/sec, lora/sec, prefix/vul, lora/vul
    # - If control=both and method=prefix: prefix/sec, prefix/vul
    # - If control=both and method=lora:   lora/sec, lora/vul
    # - If control=sec or vul: compare methods for that control (prefix then lora)
    if args.control == "both":
        for c in ["sec", "vul"]:
            for m in methods:
                plot_data.append(collect_values(scripts_dir, m, c))
                plot_labels.append(f"{m}/{c}")
        title = "HumanEval pass@100 Distribution (sec left, vul right)"
    else:
        c = args.control
        for m in methods:
            plot_data.append(collect_values(scripts_dir, m, c))
            plot_labels.append(f"{m}/{c}")
        title = f"HumanEval pass@100 Distribution"

    if all(len(d) == 0 for d in plot_data):
        print("\n[ERROR] No pass@100 values parsed. Nothing to plot.")
        return

    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path, title=title)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()

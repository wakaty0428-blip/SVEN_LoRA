#!/usr/bin/env python3
"""
box_plots_score.py

Score = (SR + FC) / 2

You can choose CONTROL = sec or vul via --control.
- For SR (security rate):
  * Prefix SR file contains both sec+vul rows -> we pick the specified control row.
  * LoRA SR files are split: ...-sec.txt / ...-vul.txt -> we load the specified suffix.

- For FC (functional correctness / HumanEval):
  * Files are split: human-eval-...-sec.txt / human-eval-...-vul.txt -> we load the specified suffix.
  * We extract ONLY pass@100.

Pairing rule:
We pair SR and FC by matching hyperparameters encoded in filenames
(lr + p/r + lm + con + kl). Then score = (sr + fc)/2 per run.

Outputs are saved under:
  <scripts_dir>/box_plots/<out>

Examples (run from scripts/):
  python box_plots_score.py --method both --control sec --out score_sec.png
  python box_plots_score.py --method both --control vul --out score_vul.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


# -----------------------
# File patterns
# -----------------------
# SR:
#   prefix: 350m-lr0.01_p16_lm0.180_con41_kl410.txt
#   lora:   350m-lr0.0001_r8_lm0.180_con41_kl410-sec.txt   (or -vul.txt)
PATTERNS_SR = {
    "prefix": ["350m-lr*_p*_lm*_con*_kl*.txt"],
    "lora": {
        "sec": ["350m-lr*_r*_lm*_con*_kl*-sec.txt"],
        "vul": ["350m-lr*_r*_lm*_con*_kl*-vul.txt"],
    },
}

# FC:
#   human-eval-350m-lr..._p..._lm..._con..._kl...-sec.txt  (or -vul.txt)
PATTERNS_FC = {
    "prefix": {
        "sec": ["human-eval-350m-lr*_p*_lm*_con*_kl*-sec.txt"],
        "vul": ["human-eval-350m-lr*_p*_lm*_con*_kl*-vul.txt"],
    },
    "lora": {
        "sec": ["human-eval-350m-lr*_r*_lm*_con*_kl*-sec.txt"],
        "vul": ["human-eval-350m-lr*_r*_lm*_con*_kl*-vul.txt"],
    },
}


# -----------------------
# Key extraction from filename
# -----------------------
# Normalize pairing key: include lr and p/r + lm + con + kl.
RE_PREFIX = re.compile(
    r"350m-lr(?P<lr>[\d.]+)_p(?P<p>\d+)_lm(?P<lm>[\d.]+)_con(?P<con>\d+)_kl(?P<kl>\d+)"
)
RE_LORA = re.compile(
    r"350m-lr(?P<lr>[\d.]+)_r(?P<r>\d+)_lm(?P<lm>[\d.]+)_con(?P<con>\d+)_kl(?P<kl>\d+)"
)


def make_key(method: str, filename: str) -> Optional[str]:
    base = Path(filename).name
    if method == "prefix":
        m = RE_PREFIX.search(base)
        if not m:
            return None
        d = m.groupdict()
        return f"lr{d['lr']}|p{d['p']}|lm{d['lm']}|con{d['con']}|kl{d['kl']}"
    else:
        m = RE_LORA.search(base)
        if not m:
            return None
        d = m.groupdict()
        return f"lr{d['lr']}|r{d['r']}|lm{d['lm']}|con{d['con']}|kl{d['kl']}"


def find_files(base: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(base.glob(pat))
    return sorted(set(files))


# -----------------------
# Parsers
# -----------------------
def parse_overall_rate(txt_path: Path, control: str) -> Optional[float]:
    """
    Parse SR from security rate table:
      | overall | overall | sec|vul | 84.0, 0.0, 100.0 | ...
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


def parse_pass_at_100(txt_path: Path) -> Optional[float]:
    """
    Parse pass@100 from HumanEval table.
    """
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

        m = re.search(r"[-+]?\d+(?:\.\d+)?", cols[pass100_col])
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


# -----------------------
# Plotting
# -----------------------
def prepare_output_path(scripts_dir: Path, out_name: str) -> Path:
    out_dir = scripts_dir / "box_plots"
    out_dir.mkdir(exist_ok=True)
    return out_dir / out_name


def make_boxplot(data: List[List[float]], labels: List[str], out_path: Path) -> None:
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Score = (SR + FC) / 2")
    plt.title("Score Distribution")
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["prefix", "lora", "both"], default="both")
    ap.add_argument("--control", choices=["sec", "vul"], default="sec",
                    help="Which control split to use for BOTH SR and FC pairing.")
    ap.add_argument("--scripts_dir", default=".")
    ap.add_argument("--out", default="boxplot_score.png")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    methods = ["prefix", "lora"] if args.method == "both" else [args.method]
    control = args.control.lower()

    plot_data: List[List[float]] = []
    plot_labels: List[str] = []

    for method in methods:
        # 1) SR map: key -> sr
        sr_map: Dict[str, float] = {}
        if method == "prefix":
            sr_files = find_files(scripts_dir, PATTERNS_SR["prefix"])
            for fp in sr_files:
                key = make_key(method, fp.name)
                if key is None:
                    continue
                sr = parse_overall_rate(fp, control=control)
                if sr is None:
                    continue
                sr_map[key] = sr
        else:
            sr_files = find_files(scripts_dir, PATTERNS_SR["lora"][control])
            for fp in sr_files:
                key = make_key(method, fp.name)
                if key is None:
                    continue
                # lora SR files should contain only that control, but parse is still safe
                sr = parse_overall_rate(fp, control=control)
                if sr is None:
                    continue
                sr_map[key] = sr

        # 2) FC map: key -> fc
        fc_map: Dict[str, float] = {}
        fc_files = find_files(scripts_dir, PATTERNS_FC[method][control])
        for fp in fc_files:
            key = make_key(method, fp.name)
            if key is None:
                continue
            fc = parse_pass_at_100(fp)
            if fc is None:
                continue
            fc_map[key] = fc

        # 3) Pair and compute score
        keys = sorted(set(sr_map.keys()) & set(fc_map.keys()))
        scores: List[float] = [(sr_map[k] + fc_map[k]) / 2.0 for k in keys]

        missing_sr = len(set(fc_map.keys()) - set(sr_map.keys()))
        missing_fc = len(set(sr_map.keys()) - set(fc_map.keys()))

        print(f"\n=== {method.upper()} ({control}) ===")
        print(f"SR files checked : {len(sr_files)} | parsed SR: {len(sr_map)}")
        print(f"FC files checked : {len(fc_files)} | parsed FC: {len(fc_map)}")
        print(f"paired runs      : {len(keys)}")
        print(f"missing SR (has FC only): {missing_sr}")
        print(f"missing FC (has SR only): {missing_fc}")
        if scores:
            mean = sum(scores) / len(scores)
            print(f"score min={min(scores):.2f}, mean={mean:.2f}, max={max(scores):.2f}")
        else:
            print("[WARN] No paired scores found. Check filename matching and availability.")

        plot_data.append(scores)
        plot_labels.append(f"{method}/{control}")


    if all(len(d) == 0 for d in plot_data):
        print("\n[ERROR] No scores parsed. Nothing to plot.")
        return

    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()

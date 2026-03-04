#!/usr/bin/env python3
"""
box_plots_fc.py

Make 3 box plots of FUNCTIONAL CORRECTNESS (HumanEval pass@100):

  1) Original
  2) Prefix
  3) LoRA

Original LM files:
  human-eval-350m-lm-trial1-seed1.txt
  human-eval-350m-lm-trial2-seed2.txt
  ...

Fine-tuned files are split by control:
  human-eval-...-sec-...
  human-eval-...-vul-...

CLI:
- --control : sec | vul
- --scripts_dir : directory containing the human-eval txt files
- --out : output PNG filename (saved under scripts_dir/box_plots/)

Example (run from scripts/):
  python box_plots_fc.py --scripts_dir 350m --control sec --out fc_sec_orig_prefix_lora.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


# =========================================================
# EDIT THESE PATTERNS FOR YOUR TARGET MODEL
# =========================================================
PATTERNS = {
    "original": {
        "any": ["human-eval-350m-lm-trial*-seed*.txt"],
    },
    "prefix": {
        "sec": ["human-eval-350m-ep7-lr0.01_p16_lm0.180_con41_kl410-sec-trial*-seed*.txt"],
        "vul": ["human-eval-350m-lr*_p*_lm*_con*_kl*-vul.txt"],
    },
    "lora": {
        "sec": ["human-eval-350m-ep7-lr1e-04_r8_a16_ld0.1_tqkv_lm0.180_con41_kl410-sec-trial*-seed*.txt"],
        "vul": ["human-eval-350m-lr*_r*_lm*_con*_kl*-vul.txt"],
    },
}


# =========================================================
# Utilities
# =========================================================

def find_txt_files(base: Path, method: str, control: str) -> List[Path]:
    files: List[Path] = []

    if method == "original":
        for pat in PATTERNS["original"]["any"]:
            files.extend(base.glob(pat))
    else:
        for pat in PATTERNS[method][control]:
            files.extend(base.glob(pat))

    return sorted(set(files))


def parse_pass_at_100(txt_path: Path) -> Optional[float]:
    """
    Parse pass@100 from HumanEval markdown table.
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

        cell = cols[pass100_col]
        m = re.search(r"[-+]?\d+(?:\.\d+)?", cell)
        if not m:
            continue

        try:
            return float(m.group(0))
        except ValueError:
            return None

    return None


def summarize(values: List[float]) -> str:
    if not values:
        return "(no values)"
    mean = sum(values) / len(values)
    return f"n={len(values)} | min={min(values):.2f} mean={mean:.2f} max={max(values):.2f}"


def prepare_output_path(scripts_dir: Path, out_name: str) -> Path:
    out_dir = scripts_dir / "box_plots"
    out_dir.mkdir(exist_ok=True)
    return out_dir / out_name


def make_boxplot(
    data: List[List[float]],
    labels: List[str],
    out_path: Path,
    title: str,
) -> None:
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Functional correctness (pass@100)")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["sec", "vul"], default="sec")
    ap.add_argument("--scripts_dir", default=".")
    ap.add_argument("--out", default="boxplot_fc.png")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    control = args.control

    # Collect values in fixed order
    original_files = find_txt_files(scripts_dir, "original", control)
    prefix_files = find_txt_files(scripts_dir, "prefix", control)
    lora_files = find_txt_files(scripts_dir, "lora", control)

    original_vals = [parse_pass_at_100(f) for f in original_files]
    prefix_vals = [parse_pass_at_100(f) for f in prefix_files]
    lora_vals = [parse_pass_at_100(f) for f in lora_files]

    original_vals = [v for v in original_vals if v is not None]
    prefix_vals = [v for v in prefix_vals if v is not None]
    lora_vals = [v for v in lora_vals if v is not None]

    print("\n=== ORIGINAL ===")
    print(summarize(original_vals))

    print(f"\n=== PREFIX/{control} ===")
    print(summarize(prefix_vals))

    print(f"\n=== LORA/{control} ===")
    print(summarize(lora_vals))

    plot_data = [original_vals, prefix_vals, lora_vals]
    plot_labels = ["Original", "Prefix", "LoRA"]

    if any(len(d) == 0 for d in plot_data):
        print("\n[ERROR] One of the groups has no values.")
        print("Fix PATTERNS or scripts_dir.")
        return

    title = f"HumanEval pass@100 Distribution"
    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path, title)

    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
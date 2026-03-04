#!/usr/bin/env python3
"""
box_plots_score.py

Score = (SR + FC) / 2

This version is aligned with your current:
- box_plots_sr.py  (Original/Prefix/LoRA, control=sec|vul)
- box_plots_fc.py  (Original/Prefix/LoRA, control=sec|vul)

It always makes EXACTLY 3 boxplots (in this order):
  1) Original
  2) Prefix
  3) LoRA

Pairing rule (same as your score script style):
- Pair SR and FC by run_id extracted from filenames.
  * If filename has "trialX-seedY" -> use that.
  * Otherwise use the final numeric suffix "-N.txt" -> treat as trialN-seedN.
- For each run_id, if multiple SR/FC values exist, pair by sorting filenames and zipping.

CLI:
- --control : sec | vul   (applies to Prefix/LoRA; Original uses orig for SR and independent for FC)
- --scripts_dir : directory containing both SR txts and HumanEval txts
- --out : output PNG filename (saved under scripts_dir/box_plots/)

Example:
  python box_plots_score.py --scripts_dir 350m --control sec --out score_sec_original_prefix_lora.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# =========================================================
# Patterns (match your latest SR/FC scripts)
# EDIT if you change experiment names
# =========================================================
PATTERNS_SR = {
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

PATTERNS_FC = {
    "original": {
        "any": ["human-eval-350m-lm-trial*-seed*.txt"],
    },
    "prefix": {
        "sec": ["human-eval-350m-ep7-lr0.01_p16_lm0.180_con41_kl410-sec-trial*-seed*.txt"],
        # If you ALSO have vul files in trial/seed format, replace this with the right glob.
        "vul": ["human-eval-350m-lr*_p*_lm*_con*_kl*-vul*.txt"],
    },
    "lora": {
        "sec": ["human-eval-350m-ep7-lr1e-04_r8_a16_ld0.1_tqkv_lm0.180_con41_kl410-sec-trial*-seed*.txt"],
        # If you ALSO have vul files in trial/seed format, replace this with the right glob.
        "vul": ["human-eval-350m-lr*_r*_lm*_con*_kl*-vul*.txt"],
    },
}


def find_files(base: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(base.glob(pat))
    return sorted(set(files))


# -----------------------
# Run-id extraction
# -----------------------
RE_TRIAL_SEED = re.compile(r"trial(?P<trial>\d+)-seed(?P<seed>\d+)", re.IGNORECASE)
RE_LAST_NUM_SUFFIX = re.compile(r"-(?P<n>\d+)\.txt$", re.IGNORECASE)


def extract_run_id(filename: str) -> Optional[str]:
    """
    Normalize to run_id "trialX-seedY".

    Priority:
      1) If filename includes "trialX-seedY" -> use it.
      2) Else use last numeric suffix "-N.txt" -> trialN-seedN.
    """
    base = Path(filename).name

    m = RE_TRIAL_SEED.search(base)
    if m:
        t = int(m.group("trial"))
        s = int(m.group("seed"))
        return f"trial{t}-seed{s}"

    m = RE_LAST_NUM_SUFFIX.search(base)
    if m:
        n = int(m.group("n"))
        return f"trial{n}-seed{n}"

    return None


# -----------------------
# Parsers
# -----------------------
def parse_overall_rate(txt_path: Path, control: str) -> Optional[float]:
    """
    Parse SR from security rate table:
      | overall | overall | sec|vul|orig | 84.0, 0.0, 100.0 | ...
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
# Grouping + pairing
# -----------------------
def group_values_by_run_id(
    files: List[Path],
    value_parser,
    control: Optional[str] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Returns: run_id -> list of (filename, value)
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    for fp in files:
        rid = extract_run_id(fp.name)
        if rid is None:
            continue

        val = value_parser(fp, control) if control is not None else value_parser(fp)
        if val is None:
            continue

        out.setdefault(rid, []).append((fp.name, float(val)))
    return out


def summarize(values: List[float]) -> str:
    if not values:
        return "(no values)"
    mean = sum(values) / len(values)
    return f"n={len(values)} | min={min(values):.2f} mean={mean:.2f} max={max(values):.2f}"


def collect_scores_for_method(scripts_dir: Path, method: str, control: str) -> List[float]:
    """
    method: "original" | "prefix" | "lora"
    control: sec|vul (used for prefix/lora)
    """
    # ---- SR files
    if method == "original":
        sr_files = find_files(scripts_dir, PATTERNS_SR["original"]["orig"])
        sr_control = "orig"
    elif method == "prefix":
        sr_files = find_files(scripts_dir, PATTERNS_SR["prefix"]["any"])
        sr_control = control
    elif method == "lora":
        sr_files = find_files(scripts_dir, PATTERNS_SR["lora"][control])
        sr_control = control
    else:
        raise ValueError(f"Unknown method: {method}")

    sr_by_run = group_values_by_run_id(
        sr_files,
        value_parser=parse_overall_rate,
        control=sr_control,
    )

    # ---- FC files
    if method == "original":
        fc_files = find_files(scripts_dir, PATTERNS_FC["original"]["any"])
    else:
        fc_files = find_files(scripts_dir, PATTERNS_FC[method][control])

    fc_by_run = group_values_by_run_id(
        fc_files,
        value_parser=parse_pass_at_100,
        control=None,
    )

    # ---- Pair by run_id
    run_ids = sorted(set(sr_by_run.keys()) & set(fc_by_run.keys()))
    scores: List[float] = []

    for rid in run_ids:
        sr_list = sorted(sr_by_run[rid], key=lambda x: x[0])
        fc_list = sorted(fc_by_run[rid], key=lambda x: x[0])

        if len(sr_list) != len(fc_list):
            print(f"[WARN] {method}/{control} {rid}: SR count={len(sr_list)} != FC count={len(fc_list)} (zip pairing)")

        for (sr_name, sr_val), (fc_name, fc_val) in zip(sr_list, fc_list):
            scores.append((sr_val + fc_val) / 2.0)

    # ---- Debug summary
    tag = "ORIGINAL" if method == "original" else f"{method.upper()}/{control}"
    print(f"\n=== {tag} ===")
    print(f"SR files checked : {len(sr_files)} | parsed run_ids: {len(sr_by_run)}")
    print(f"FC files checked : {len(fc_files)} | parsed run_ids: {len(fc_by_run)}")
    print(f"paired run_ids   : {len(run_ids)}")
    print(f"paired samples   : {len(scores)}")
    print(f"scores           : {summarize(scores)}")

    only_sr = sorted(set(sr_by_run.keys()) - set(fc_by_run.keys()))
    only_fc = sorted(set(fc_by_run.keys()) - set(sr_by_run.keys()))
    if only_sr:
        print(f"[INFO] SR-only run_ids (no FC): {only_sr[:10]}")
    if only_fc:
        print(f"[INFO] FC-only run_ids (no SR): {only_fc[:10]}")

    return scores


# -----------------------
# Plotting
# -----------------------
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
    plt.ylabel("Score = (SR + FC) / 2")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", choices=["sec", "vul"], default="sec",
                    help="Which control split to use for Prefix/LoRA (Original is independent).")
    ap.add_argument("--scripts_dir", default=".")
    ap.add_argument("--out", default="boxplot_score.png")
    args = ap.parse_args()

    scripts_dir = Path(args.scripts_dir).resolve()
    control = args.control.lower()

    # Always 3 groups, in fixed order
    scores_original = collect_scores_for_method(scripts_dir, "original", control)
    scores_prefix = collect_scores_for_method(scripts_dir, "prefix", control)
    scores_lora = collect_scores_for_method(scripts_dir, "lora", control)

    plot_data = [scores_original, scores_prefix, scores_lora]
    plot_labels = ["Original", "Prefix", "LoRA"]

    if any(len(d) == 0 for d in plot_data):
        print("\n[ERROR] One of the groups has no paired scores.")
        print("Fix PATTERNS_* and/or ensure SR+FC filenames share run_id (trial/seed or -N suffix).")
        return

    title = "Score Distribution"
    out_path = prepare_output_path(scripts_dir, args.out)
    make_boxplot(plot_data, plot_labels, out_path, title=title)
    print(f"\nSaved plot → {out_path}")


if __name__ == "__main__":
    main()
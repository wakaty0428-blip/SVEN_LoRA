#!/usr/bin/env python3
# ==============================================================
# grid_train_prefix.py
#
# Purpose
#   Train Prefix-Tuning models for CodeGen using either:
#     (A) an explicit list of run names (SELECTED_RUNS), or
#     (B) a full Cartesian-product grid (if SELECTED_RUNS is empty).
#
# Two modes
#   1) SELECTED_RUNS mode (recommended for controlled experiments)
#      - Put run names in SELECTED_RUNS.
#      - Each run name encodes hyperparameters (lr, p, lm, con, kl).
#      - The script parses those values and calls train.py for each run.
#
#   2) Full grid mode
#      - Set SELECTED_RUNS = [].
#      - Define the hyperparameter lists (learning_rates, prefix_tokens,
#        lm_loss_ratios, contrastive_ratios, kl_ratios).
#      - The script generates all combinations via itertools.product().
#
# Run naming convention (required)
#   <base>-lr<lr>_p<p>_lm<lm>_con<con>_kl<kl>
#
# Example
#   6b-lr0.01_p8_lm0.180_con41_kl390
#
# Meaning of fields
#   - lr  : learning rate passed to train.py
#   - p   : number of prefix tokens (n_prefix_token) a.k.a. prefix length
#   - lm  : lm_loss_ratio (float) passed directly to train.py
#   - con : contrastive_loss_ratio (int); trainer interprets as con/100
#   - kl  : kl_loss_ratio (int); trainer interprets as kl/1000
#
# What this script does NOT do (unlike your LoRA grid script)
#   - It does not stream output to train.log
#   - It does not skip already-finished runs
#   - It does not clean up checkpoint-epoch-* directories

import itertools
import subprocess
import re


# ==============================================================
# 1) Manually specify runs you want to train (optional)
#    If empty, script runs full grid below.
# ==============================================================
SELECTED_RUNS = [
    # Examples:
    "6b-lr0.01_p8_lm0.220_con41_kl370",
    "6b-lr0.01_p8_lm0.300_con29_kl410",
    "6b-lr0.01_p8_lm0.400_con31_kl290",
    "6b-lr0.01_p8_lm0.500_con25_kl250",
]

# ==============================================================
# 2) Hyperparameter grids (used only when SELECTED_RUNS == [])
# ==============================================================
# learning_rates = [1e-2]            # e.g., [1e-2, 5e-3]
# prefix_tokens = 8            # prefix length p
# lm_loss_ratios = [0.180]           # raw lm ratio passed to train.py
# contrastive_ratios = [35, 39, 41]  # raw con ratio (SVEN: /100 in trainer)
# kl_ratios = [350, 390, 410]        # raw kl ratio (SVEN: /1000 in trainer)

# ==============================================================
# 3) Base settings
# ==============================================================
pretrain = "Salesforce/codegen-6B-multi"
model_type = "prefix"
base = "6b"
num_train_epochs = 5

# ==============================================================
# 4) Parse run name
# ==============================================================
# Accepts:
#   350m-lr0.01_p16_lm0.180_con41_kl390
RUN_PATTERN = re.compile(
    r"^(?P<base>[^-]+)-"
    r"lr(?P<lr>[\d.eE+-]+)_"
    r"p(?P<p>\d+)_"
    r"lm(?P<lm>[\d.]+)_"
    r"con(?P<con>\d+)_"
    r"kl(?P<kl>\d+)$"
)

def parse_run_name(run_name: str):
    m = RUN_PATTERN.match(run_name)
    if not m:
        raise ValueError(
            f"Run name does not match expected pattern:\n"
            f"  {run_name}\n\n"
            f"Expected format:\n"
            f"  {base}-lr<lr>_p<p>_lm<lm>_con<con>_kl<kl>\n"
            f"Example:\n"
            f"  6b-lr0.01_p8_lm0.180_con41_kl390"
        )
    return {
        "lr": float(m.group("lr")),
        "p": int(m.group("p")),
        "lm": float(m.group("lm")),
        "con": int(m.group("con")),
        "kl": int(m.group("kl")),
    }

# ==============================================================
# 5) Build run list
# ==============================================================
run_list = []

if SELECTED_RUNS:
    for name in SELECTED_RUNS:
        hp = parse_run_name(name)
        run_list.append((name, hp))
else:
    for lr, p, lm, con, kl in itertools.product(
        learning_rates, prefix_tokens, lm_loss_ratios, contrastive_ratios, kl_ratios
    ):
        name = f"{base}-lr{lr}_p{p}_lm{lm:.3f}_con{con}_kl{kl}"
        hp = {"lr": lr, "p": p, "lm": lm, "con": con, "kl": kl}
        run_list.append((name, hp))

# ==============================================================
# 6) Execute
# ==============================================================
for run_name, hp in run_list:
    cmd = [
        "python", "train.py",
        "--output_name", run_name,
        "--model_type", model_type,
        "--pretrain_dir", pretrain,
        "--learning_rate", str(hp["lr"]),
        "--n_prefix_token", str(hp["p"]),              # <-- prefix length
        "--lm_loss_ratio", str(hp["lm"]),
        "--contrastive_loss_ratio", str(hp["con"]),
        "--kl_loss_ratio", str(hp["kl"]),
        "--num_train_epochs", str(num_train_epochs),
    ]

    print("===================================================")
    print(f"▶ Running experiment: {run_name}")
    print("Parsed hyperparameters:", hp)
    print("Command:", " ".join(cmd))
    print("Saving into:", f"../trained/{run_name}")
    print("===================================================\n")

    subprocess.run(cmd, check=False)

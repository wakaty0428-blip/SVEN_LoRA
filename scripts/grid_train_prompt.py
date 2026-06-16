#!/usr/bin/env python3
# ==============================================================
# grid_train_prompt.py
#
# Purpose
#   Train Prompt-Tuning models for CodeGen using either:
#     (A) an explicit list of run names (SELECTED_RUNS), or
#     (B) a full Cartesian-product grid (if SELECTED_RUNS is empty).
#
# Two modes
#   1) SELECTED_RUNS mode (recommended for controlled experiments)
#      - Put run names in SELECTED_RUNS.
#      - Each run name encodes hyperparameters (lr, v, lm, con, kl).
#      - The script parses those values and calls train.py for each run.
#
#   2) Full grid mode
#      - Set SELECTED_RUNS = [].
#      - Define the hyperparameter lists (learning_rates, virtual_tokens,
#        lm_loss_ratios, contrastive_ratios, kl_ratios).
#      - The script generates all combinations via itertools.product().
#
# Run naming convention (required)
#   <base>-lr<lr>_v<v>_lm<lm>_con<con>_kl<kl>
#
# Example
#   350m-lr0.05_v50_lm0.180_con41_kl390
#
# Meaning of fields
#   - lr  : learning rate passed to train.py
#   - v   : number of virtual tokens (n_virtual_token) a.k.a. prompt length
#   - lm  : lm_loss_ratio (float) passed directly to train.py
#   - con : contrastive_loss_ratio (int); trainer interprets as con/100
#   - kl  : kl_loss_ratio (int); trainer interprets as kl/1000
#
# Notes on prompt-tuning hyperparameters (vs. prefix-tuning)
#   - Virtual tokens replace prefix tokens; typical range is 20-100
#     (vs. 10-30 for prefix-tuning) per Table 7 of the paper.
#   - Learning rate is higher (1e-2 to 1e-1) because continuous prompt
#     embeddings require more aggressive optimization than prefix vectors.
#   - model_type is "prompt" (not "prefix").
#   - The CLI flag is --n_virtual_token (not --n_prefix_token).

import itertools
import subprocess
import re


# ==============================================================
# 1) Manually specify runs you want to train (optional)
#    If empty, script runs full grid below.
# ==============================================================
SELECTED_RUNS = [
    # Examples:
    "350m-lr0.05_v10_lm0.180_con41_kl410",
    "350m-lr0.05_v20_lm0.180_con41_kl410",
    "350m-lr0.05_v30_lm0.180_con41_kl410",
    "350m-lr0.05_v40_lm0.180_con41_kl410",
    "350m-lr0.05_v50_lm0.180_con41_kl410",
]

# ==============================================================
# 2) Hyperparameter grids (used only when SELECTED_RUNS == [])
# ==============================================================
# learning_rates     = [5e-2]            # prompt-tuning: 1e-2 to 1e-1
# n_prompt_token     = [50]              # prompt length v (20-100 for 6-7B)
# lm_loss_ratios     = [0.180]           # raw lm ratio passed to train.py
# contrastive_ratios = [35, 39, 41]      # raw con ratio (SVEN: /100 in trainer)
# kl_ratios          = [350, 390, 410]   # raw kl ratio (SVEN: /1000 in trainer)

# ==============================================================
# 3) Base settings
# ==============================================================
pretrain = "Salesforce/codegen-350M-multi"
model_type = "prompt"
base = "350m"
num_train_epochs = 7

# ==============================================================
# 4) Parse run name
# ==============================================================
# Accepts:
#   350m-lr0.05_v50_lm0.180_con41_kl390
RUN_PATTERN = re.compile(
    r"^(?P<base>[^-]+)-"
    r"lr(?P<lr>[\d.eE+-]+)_"
    r"v(?P<v>\d+)_"
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
            f"  {base}-lr<lr>_v<v>_lm<lm>_con<con>_kl<kl>\n"
            f"Example:\n"
            f"  350m-lr0.05_v50_lm0.180_con41_kl390"
        )
    return {
        "lr": float(m.group("lr")),
        "v": int(m.group("v")),
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
    for lr, v, lm, con, kl in itertools.product(
        learning_rates, virtual_tokens, lm_loss_ratios, contrastive_ratios, kl_ratios
    ):
        name = f"{base}-lr{lr}_v{v}_lm{lm:.3f}_con{con}_kl{kl}"
        hp = {"lr": lr, "v": v, "lm": lm, "con": con, "kl": kl}
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
        "--n_prompt_token", str(hp["v"]),              # <-- prompt length
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
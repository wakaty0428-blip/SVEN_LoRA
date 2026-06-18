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
#      - Each run name encodes hyperparameters (lr, v, lm, con, kl, ep).
#      - The script parses those values and calls train.py for each run.
#
#   2) Full grid mode
#      - Set SELECTED_RUNS = [].
#      - Define the hyperparameter lists (learning_rates, virtual_tokens,
#        lm_loss_ratios, contrastive_ratios, kl_ratios).
#      - The script generates all combinations via itertools.product().
#
# Run naming convention (required)
#   <base>-lr<lr>_v<v>_lm<lm>_con<con>_kl<kl>_ep<ep>
#
# Example
#   350m-lr0.05_v50_lm0.180_con41_kl390_ep20
#
# Meaning of fields
#   - lr  : learning rate passed to train.py
#   - v   : number of virtual tokens (n_virtual_token) a.k.a. prompt length
#   - lm  : lm_loss_ratio (float) passed directly to train.py
#   - con : contrastive_loss_ratio (int); trainer interprets as con/100
#   - kl  : kl_loss_ratio (int); trainer interprets as kl/1000
#   - ep  : num_train_epochs (int) used for training
#
# Disk management
#   After each run, cleanup_epoch_checkpoints() deletes intermediate
#   epoch checkpoints (checkpoint-epoch-*), keeping ONLY the final
#   (highest-numbered) epoch checkpoint, plus checkpoint-last and any
#   non-checkpoint files (e.g. train.log).
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
from pathlib import Path


# ==============================================================
# 1) Manually specify runs you want to train (optional)
#    If empty, script runs full grid below.
# ==============================================================
SELECTED_RUNS = [
    # Examples:
    "350m-lr0.05_v1_lm0.180_con41_kl410_ep7",
    "350m-lr0.05_v5_lm0.180_con41_kl410_ep7",
    "350m-lr0.05_v20_lm0.180_con41_kl410_ep7",
    "350m-lr0.05_v100_lm0.180_con41_kl410_ep7",
    "350m-lr0.05_v150_lm0.180_con41_kl410_ep7",
]

# ==============================================================
# 2) Hyperparameter grids (used only when SELECTED_RUNS == [])
# ==============================================================
# learning_rates     = [5e-2]            # prompt-tuning: 1e-2 to 1e-1
# virtual_tokens     = [50]              # prompt length v (20-100 for 6-7B)
# lm_loss_ratios     = [0.180]           # raw lm ratio passed to train.py
# contrastive_ratios = [35, 39, 41]      # raw con ratio (SVEN: /100 in trainer)
# kl_ratios          = [350, 390, 410]   # raw kl ratio (SVEN: /1000 in trainer)

# ==============================================================
# 3) Base settings
# ==============================================================
pretrain = "Salesforce/codegen-350M-multi"
model_type = "prompt"
base = "350m"
num_train_epochs = 7          # default; used when run name omits _ep<ep>

# where train.py writes runs (relative to /scripts)
TRAINED_ROOT = Path("../trained")

# ==============================================================
# 4) Parse run name
# ==============================================================
# Accepts (ep is optional for backward compatibility):
#   350m-lr0.05_v50_lm0.180_con41_kl390_ep20
#   350m-lr0.05_v50_lm0.180_con41_kl390      (ep falls back to num_train_epochs)
RUN_PATTERN = re.compile(
    r"^(?P<base>[^-]+)-"
    r"lr(?P<lr>[\d.eE+-]+)_"
    r"v(?P<v>\d+)_"
    r"lm(?P<lm>[\d.]+)_"
    r"con(?P<con>\d+)_"
    r"kl(?P<kl>\d+)"
    r"(?:_ep(?P<ep>\d+))?$"
)

def parse_run_name(run_name: str):
    m = RUN_PATTERN.match(run_name)
    if not m:
        raise ValueError(
            f"Run name does not match expected pattern:\n"
            f"  {run_name}\n\n"
            f"Expected format:\n"
            f"  {base}-lr<lr>_v<v>_lm<lm>_con<con>_kl<kl>_ep<ep>\n"
            f"Example:\n"
            f"  350m-lr0.05_v50_lm0.180_con41_kl390_ep20"
        )
    ep = m.group("ep")
    return {
        "lr": float(m.group("lr")),
        "v": int(m.group("v")),
        "lm": float(m.group("lm")),
        "con": int(m.group("con")),
        "kl": int(m.group("kl")),
        "ep": int(ep) if ep is not None else num_train_epochs,
    }

# ==============================================================
# Cleanup (keep ONLY the final epoch checkpoint)
# ==============================================================
def cleanup_epoch_checkpoints(run_dir: Path) -> None:
    """
    Delete intermediate epoch checkpoints, keeping ONLY the final epoch.

    Keeps:
        - checkpoint-epoch-<MAX>   (the highest-numbered epoch = final epoch)
        - checkpoint-last          (if present)
        - train.log                (and any non checkpoint-epoch-* files)

    Deletes:
        - all other checkpoint-epoch-* directories (intermediate epochs)
    """
    if not run_dir.exists():
        return

    epoch_re = re.compile(r"^checkpoint-epoch-(\d+)$")

    # Collect (epoch_number, path) for every epoch checkpoint dir.
    epoch_ckpts = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = epoch_re.match(p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))

    if not epoch_ckpts:
        return

    # Identify the final (highest) epoch checkpoint to keep.
    final_epoch, final_path = max(epoch_ckpts, key=lambda x: x[0])

    removed = 0
    for epoch_num, p in epoch_ckpts:
        if p == final_path:
            continue  # keep the final epoch
        subprocess.run(["rm", "-rf", str(p)], check=False)
        removed += 1

    if removed:
        print(
            f"[CLEANUP] Removed {removed} intermediate epoch checkpoints under "
            f"{run_dir} (kept checkpoint-epoch-{final_epoch} as final)\n"
        )
    else:
        print(
            f"[CLEANUP] Only the final epoch checkpoint "
            f"(checkpoint-epoch-{final_epoch}) present under {run_dir}; nothing removed\n"
        )

# ==============================================================
# 5) Build run list
# ==============================================================
run_list = []

if SELECTED_RUNS:
    for name in SELECTED_RUNS:
        hp = parse_run_name(name)
        # Ensure the name always carries the epoch field for clarity.
        if not name.endswith(f"_ep{hp['ep']}"):
            name = f"{name}_ep{hp['ep']}"
        run_list.append((name, hp))
else:
    for lr, v, lm, con, kl in itertools.product(
        learning_rate, n_prompt_token, lm_loss_ratio, contrastive_loss_ratio, kl_loss_ratio
    ):
        name = f"{base}-lr{lr}_v{v}_lm{lm:.3f}_con{con}_kl{kl}_ep{num_train_epochs}"
        hp = {"lr": lr, "v": v, "lm": lm, "con": con, "kl": kl, "ep": num_train_epochs}
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
        "--num_train_epochs", str(hp["ep"]),           # <-- epochs from run name
    ]

    print("===================================================")
    print(f"▶ Running experiment: {run_name}")
    print("Parsed hyperparameters:", hp)
    print("Command:", " ".join(cmd))
    print("Saving into:", f"../trained/{run_name}")
    print("===================================================\n")

    subprocess.run(cmd, check=False)

    # Keep only the final epoch checkpoint; delete intermediate ones.
    cleanup_epoch_checkpoints(TRAINED_ROOT / run_name)
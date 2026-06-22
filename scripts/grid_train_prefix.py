#!/usr/bin/env python3
# ==============================================================
# grid_train_prefix.py
#
# Purpose
#   Train prefix-tuning models for CodeGen using either:
#     (A) an explicit list of run names (SELECTED_RUNS), or
#     (B) a full Cartesian-product grid (if SELECTED_RUNS is empty).
#
# Two modes
#   1) SELECTED_RUNS mode (recommended for controlled experiments)
#      - Put run names in SELECTED_RUNS.
#      - Each run name encodes hyperparameters (lr, p, lm, con, kl, ep).
#      - The script parses those values and calls train.py for each run.
#
#   2) Full grid mode
#      - Set SELECTED_RUNS = [].
#      - Define the hyperparameter lists (learning_rate, n_prefix_token,
#        lm_loss_ratios, contrastive_ratios, kl_ratios).
#      - The script generates all combinations via itertools.product().
#
# Run naming convention (required)
#   <base>-lr<lr>_p<p>_lm<lm>_con<con>_kl<kl>_ep<ep>
#
# Example
#   350m-lr0.05_p50_lm0.180_con41_kl390_ep20
#
# Meaning of fields
#   - lr  : learning rate passed to train.py
#   - p   : number of prefix tokens (n_prefix_token) a.k.a. prefix length
#   - lm  : lm_loss_ratio (float) passed directly to train.py
#   - con : contrastive_loss_ratio (int); trainer interprets as con/100
#   - kl  : kl_loss_ratio (int); trainer interprets as kl/1000
#   - ep  : num_train_epochs (int) used for training
#
# Resume / skip behavior
#   If SKIP_EXISTING is True, any run whose output directory already
#   contains a COMPLETED training run is skipped (not retrained).
#   "Completed" means the run dir exists AND holds at least one final-state
#   checkpoint (checkpoint-epoch-* or checkpoint-last). A bare/empty dir
#   from a crashed run is NOT treated as done, so it will be retrained.
#
# Disk management
#   After each run, cleanup_epoch_checkpoints() deletes intermediate
#   epoch checkpoints (checkpoint-epoch-*), keeping ONLY the final
#   (highest-numbered) epoch checkpoint, plus checkpoint-last and any
#   non-checkpoint files (e.g. train.log).
#
# Notes on prefix-tuning hyperparameters
#   - Prefix tokens range typically between 20-100 per Table 7 of the paper.
#   - Learning rate is higher (1e-2 to 1e-1) because continuous prefix
#     embeddings require more aggressive optimization than standard finetuning.
#   - model_type is "prefix".
#   - The CLI flag is --n_prefix_token.

import itertools
import subprocess
import re
from pathlib import Path


# ==============================================================
# 1) Manually specify runs you want to train (optional)
#    If empty, script runs full grid below.
# ==============================================================
SELECTED_RUNS = []

# ==============================================================
# 2) Hyperparameter grids (used only when SELECTED_RUNS == [])
# ==============================================================
learning_rate         = [0.01]       # prefix-tuning scales: 1e-2 to 1e-1
n_prefix_token         = [16]         # prefix length candidates (p)
lm_loss_ratio         = [0.180]      # raw lm ratio passed to train.py
contrastive_loss_ratio = [41]         # raw con ratio (SVEN: /100 in trainer)
kl_loss_ratio          = [410]        # raw kl ratio (SVEN: /1000 in trainer)

# ==============================================================
# 3) Base settings
# ==============================================================
pretrain = "Salesforce/codegen-350M-multi"
model_type = "prefix"
base = "350m"
num_train_epochs = 7          # default; used when run name omits _ep<ep>

# Skip a run if a completed training output already exists for its name.
# Set to False to always (re)train every run regardless of existing output.
SKIP_EXISTING = True

# where train.py writes runs (relative to /scripts)
TRAINED_ROOT = Path("../trained")

# ==============================================================
# 4) Parse run name
# ==============================================================
# Accepts (ep is optional for backward compatibility):
#   350m-lr0.05_p50_lm0.180_con41_kl390_ep20
#   350m-lr0.05_p50_lm0.180_con41_kl390      (ep falls back to num_train_epochs)
RUN_PATTERN = re.compile(
    r"^(?P<base>[^-]+)-"
    r"lr(?P<lr>[\d.eE+-]+)_"
    r"p(?P<p>\d+)_"
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
            f"  {base}-lr<lr>_p<p>_lm<lm>_con<con>_kl<kl>_ep<ep>\n"
            f"Example:\n"
            f"  350m-lr0.05_p50_lm0.180_con41_kl390_ep20"
        )
    ep = m.group("ep")
    return {
        "lr": float(m.group("lr")),
        "p": int(m.group("p")),
        "lm": float(m.group("lm")),
        "con": int(m.group("con")),
        "kl": int(m.group("kl")),
        "ep": int(ep) if ep is not None else num_train_epochs,
    }

# ==============================================================
# Skip check (has this run already been trained?)
# ==============================================================
def run_already_completed(run_dir: Path) -> bool:
    """
    Return True if a run directory already holds a COMPLETED training run,
    so it can be safely skipped.
    """
    if not run_dir.exists():
        return False

    epoch_re = re.compile(r"^checkpoint-epoch-\d+$")
    has_epoch_ckpt = any(
        p.is_dir() and epoch_re.match(p.name)
        for p in run_dir.iterdir()
    )
    has_last_ckpt = (run_dir / "checkpoint-last").exists()
    return has_epoch_ckpt or has_last_ckpt

# ==============================================================
# Cleanup (keep ONLY the final epoch checkpoint)
# ==============================================================
def cleanup_epoch_checkpoints(run_dir: Path) -> None:
    """
    Delete intermediate epoch checkpoints, keeping ONLY the final epoch.
    """
    if not run_dir.exists():
        return

    epoch_re = re.compile(r"^checkpoint-epoch-(\d+)$")

    epoch_ckpts = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = epoch_re.match(p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))

    if not epoch_ckpts:
        return

    final_epoch, final_path = max(epoch_ckpts, key=lambda x: x[0])

    removed = 0
    for epoch_num, p in epoch_ckpts:
        if p == final_path:
            continue  
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
        if not name.endswith(f"_ep{hp['ep']}"):
            name = f"{name}_ep{hp['ep']}"
        run_list.append((name, hp))
else:
    for lr, p, lm, con, kl in itertools.product(
        learning_rate, n_prefix_token, lm_loss_ratio, contrastive_loss_ratio, kl_loss_ratio
    ):
        name = f"{base}-lr{lr}_p{p}_lm{lm:.3f}_con{con}_kl{kl}_ep{num_train_epochs}"
        hp = {"lr": lr, "p": p, "lm": lm, "con": con, "kl": kl, "ep": num_train_epochs}
        run_list.append((name, hp))

# ==============================================================
# 6) Execute
# ==============================================================
skipped = 0
trained = 0

for run_name, hp in run_list:
    run_dir = TRAINED_ROOT / run_name

    if SKIP_EXISTING and run_already_completed(run_dir):
        print("===================================================")
        print(f"⏭  Skipping (already trained): {run_name}")
        print(f"   Found existing run at: {run_dir}")
        print("===================================================\n")
        skipped += 1
        continue

    cmd = [
        "python", "train.py",
        "--output_name", run_name,
        "--model_type", model_type,
        "--pretrain_dir", pretrain,
        "--learning_rate", str(hp["lr"]),
        "--n_prefix_token", str(hp["p"]),               # <-- prefix length (p)
        "--lm_loss_ratio", str(hp["lm"]),
        "--contrastive_loss_ratio", str(hp["con"]),
        "--kl_loss_ratio", str(hp["kl"]),
        "--num_train_epochs", str(hp["ep"]),
    ]

    print("===================================================")
    print(f"▶ Running experiment: {run_name}")
    print("Parsed hyperparameters:", hp)
    print("Command:", " ".join(cmd))
    print("Saving into:", f"../trained/{run_name}")
    print("===================================================\n")

    subprocess.run(cmd, check=False)
    trained += 1

    cleanup_epoch_checkpoints(run_dir)

print("===================================================")
print(f"Done. Trained: {trained} | Skipped (already existed): {skipped} | "
      f"Total: {len(run_list)}")
print("===================================================")
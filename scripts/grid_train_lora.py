#!/usr/bin/env python3
# grid_train_lora.py
#
# Purpose
#   Run a simple grid search for LoRA fine-tuning runs on a fixed base model
#   (Salesforce/codegen-6B-multi), while keeping the multi-objective loss weights
#   fixed across all runs.
#
# What this script does
#   1) Enumerates a grid over LoRA hyperparameters:
#        - learning rate (LEARNING_RATES)
#        - target modules (LORA_TARGETS), e.g. "qkv_proj" or "out_proj"
#        - rank r (LORA_RS)
#        - dropout (LORA_DROPOUTS)
#        - warmup steps (WARMUP_STEPS_LIST)
#        - gradient accumulation steps (GRAD_ACC_STEPS_LIST or --grad_acc_steps)
#
#   2) Uses ONE fixed multi-objective weighting setting for all runs:
#        lm_loss_ratio + contrastive_loss_ratio/100 + kl_loss_ratio/1000 = 1
#      where contrastive_loss_ratio and kl_loss_ratio are stored as integers
#      (like the original SVEN-style convention), and lm_loss_ratio is computed
#      so the total sums to 1.
#
#   3) For each configuration, constructs a unique output_name (run directory name)
#      encoding all hyperparameters and fixed loss weights, then calls:
#        python train.py --output_name <run_name> ... (LoRA args + weights + schedule)
#
# Logging / reproducibility features
#   - Streams training stdout+stderr to BOTH:
#        (a) the terminal, and
#        (b) ../trained/<run_name>/train.log
#   - Skips a run if ../trained/<run_name>/train.log already exists
#     (useful when resuming an interrupted grid).
#
# Disk-saving cleanup
#   - After each run, deletes checkpoint-epoch-* directories under the run folder
#     to save disk space, while keeping:
#        - checkpoint-last
#        - train.log

import os
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict

# ==============================================================
# Base settings
# ==============================================================
PRETRAIN = "Salesforce/codegen-6B-multi"
MODEL_TYPE = "lora"
BASE = "6b"
NUM_TRAIN_EPOCHS = 5
TRAINED_ROOT = Path("../trained")

# ==============================================================
# Fixed loss weights (ONE setting; sum must be 1)
# ==============================================================
FIXED_CONTRASTIVE_RATIO = 41   # -> 0.30
FIXED_KL_RATIO = 410           # -> 0.40
FIXED_LM_RATIO = 1.0 - (FIXED_CONTRASTIVE_RATIO / 100.0) - (FIXED_KL_RATIO / 1000.0)

if FIXED_LM_RATIO < 0:
    raise ValueError(
        f"Invalid fixed weights: lm={FIXED_LM_RATIO:.4f} "
        f"(con={FIXED_CONTRASTIVE_RATIO}, kl={FIXED_KL_RATIO})"
    )

# ==============================================================
# Grid (EDIT THESE FREELY)
# ==============================================================
LEARNING_RATES = [1e-04]
LORA_TARGETS = ["qkv_proj", "out_proj"]          # e.g. ["out_proj", "qkv_proj,out_proj"]
LORA_RS = [4, 8]
LORA_DROPOUTS = [0.1]
WARMUP_STEPS_LIST = [0]                         # e.g. [0, 200]

# ✅ alpha is NOT fixed anymore:
#    we will set alpha = 2 * r inside build_runs()

# NEW: optionally sweep accumulation steps here
# If you pass --grad_acc_steps on CLI, that overrides this list.
GRAD_ACC_STEPS_LIST = [2]


# ==============================================================
# Helpers
# ==============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grad_acc_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps to pass to train.py. "
             "If omitted, uses GRAD_ACC_STEPS_LIST in the script."
    )
    p.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help='Optional: set CUDA_VISIBLE_DEVICES, e.g. "0" or "0,1,2,3". '
             "If omitted, leaves environment unchanged."
    )
    return p.parse_args()


def visible_gpu_count() -> int:
    """
    Best-effort count from CUDA_VISIBLE_DEVICES.
    If not set, return 1 (unknown). This is only for printing effective batch.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return 1
    return len([x for x in cvd.split(",") if x.strip() != ""])


def build_run_name(hp: Dict) -> str:
    tgt_tag = hp["lora_target_modules"].replace(",", "+")
    lr_tag = str(hp["learning_rate"])
    return (
        f"{BASE}-ep{NUM_TRAIN_EPOCHS}"
        f"-lr{lr_tag}"
        f"_r{hp['lora_r']}_a{hp['lora_alpha']}_ld{hp['lora_dropout']}"
        f"_t{tgt_tag}"
        f"_wu{hp['warmup_steps']}"
        f"_ga{hp['grad_acc_steps']}"
        f"_lm{hp['lm_loss_ratio']:.3f}_con{hp['contrastive_loss_ratio']}_kl{hp['kl_loss_ratio']}"
    )


def cleanup_epoch_checkpoints(run_dir: Path) -> None:
    """
    Delete checkpoint-epoch-* directories to save disk, but keep:
      - checkpoint-last
      - train.log
    """
    if not run_dir.exists():
        return

    removed = 0
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-epoch-"):
            subprocess.run(["rm", "-rf", str(p)], check=False)
            removed += 1

    if removed > 0:
        print(f"[CLEANUP] Removed {removed} epoch checkpoints under {run_dir}\n")


def run_train_and_log(cmd: List[str], log_path: Path) -> None:
    """
    Run train.py, streaming stdout to both terminal and train.log.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def build_runs(grad_acc_steps_values: List[int]) -> List[Dict]:
    runs: List[Dict] = []
    for lr in LEARNING_RATES:
        for tgt in LORA_TARGETS:
            for r in LORA_RS:
                lora_alpha = r       # alpha = rank
                # lora_alpha = 2 * r  # ✅ alpha = 2 * rank
                # lora_alpha = 4 * r
                for ld in LORA_DROPOUTS:
                    for wu in WARMUP_STEPS_LIST:
                        for ga in grad_acc_steps_values:
                            runs.append({
                                "learning_rate": lr,
                                "lora_target_modules": tgt,
                                "lora_r": r,
                                "lora_alpha": lora_alpha,  # ✅ now depends on r
                                "lora_dropout": ld,
                                "warmup_steps": wu,
                                "grad_acc_steps": ga,

                                # fixed weights (sum=1)
                                "contrastive_loss_ratio": FIXED_CONTRASTIVE_RATIO,
                                "kl_loss_ratio": FIXED_KL_RATIO,
                                "lm_loss_ratio": FIXED_LM_RATIO,
                            })
    return runs


# ==============================================================
# Main
# ==============================================================

def main() -> None:
    args = parse_args()

    # Optional: set visible GPUs from CLI
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    # Determine which grad_acc_steps values to use
    if args.grad_acc_steps is not None:
        if args.grad_acc_steps < 1:
            raise ValueError("--grad_acc_steps must be >= 1")
        grad_acc_steps_values = [args.grad_acc_steps]
    else:
        grad_acc_steps_values = GRAD_ACC_STEPS_LIST

    runs = build_runs(grad_acc_steps_values)

    per_gpu_batch = 1
    ngpu = visible_gpu_count()

    print("===================================================")
    print("Grid: LoRA hyperparams with ONE fixed weight setting")
    print(f"Base model    : {PRETRAIN}")
    print(f"Model type    : {MODEL_TYPE}")
    print(f"Epochs        : {NUM_TRAIN_EPOCHS}")
    print("Fixed weights : "
          f"lm={FIXED_LM_RATIO:.3f}, "
          f"ct={FIXED_CONTRASTIVE_RATIO/100.0:.3f}, "
          f"kl={FIXED_KL_RATIO/1000.0:.3f}  (sum=1.000)")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"Assumed per-GPU batch: {per_gpu_batch} (for effective-batch print)")
    print(f"Detected n_gpu (from CVD): {ngpu}")
    print(f"grad_acc_steps candidates: {grad_acc_steps_values}")
    print(f"Total runs    : {len(runs)}")
    print("===================================================\n")

    for i, hp in enumerate(runs, start=1):
        run_name = build_run_name(hp)
        run_dir = TRAINED_ROOT / run_name
        log_path = run_dir / "train.log"

        eff_batch = per_gpu_batch * ngpu * hp["grad_acc_steps"]

        cmd = [
            "python", "train.py",
            "--output_name", run_name,
            "--model_type", MODEL_TYPE,
            "--pretrain_dir", PRETRAIN,
            "--learning_rate", str(hp["learning_rate"]),

            # LoRA params
            "--lora_r", str(hp["lora_r"]),
            "--lora_alpha", str(hp["lora_alpha"]),
            "--lora_dropout", str(hp["lora_dropout"]),
            "--lora_target_modules", hp["lora_target_modules"],

            # fixed objective weights (sum=1)
            "--contrastive_loss_ratio", str(hp["contrastive_loss_ratio"]),
            "--kl_loss_ratio", str(hp["kl_loss_ratio"]),
            "--lm_loss_ratio", str(hp["lm_loss_ratio"]),

            # schedule / training
            "--warmup_steps", str(hp["warmup_steps"]),
            "--num_train_epochs", str(NUM_TRAIN_EPOCHS),

            # accumulation
            "--grad_acc_steps", str(hp["grad_acc_steps"]),
        ]

        print("===================================================")
        print(f"[{i:02d}/{len(runs)}] ▶ Running experiment: {run_name}")
        print("HP:",
              f"lr={hp['learning_rate']}, r={hp['lora_r']}, alpha={hp['lora_alpha']}, "
              f"dropout={hp['lora_dropout']}, target={hp['lora_target_modules']}, "
              f"warmup={hp['warmup_steps']}, grad_acc_steps={hp['grad_acc_steps']}")
        print(f"Effective global batch per update (assuming per_gpu=1): {eff_batch}")
        print("Command:", " ".join(cmd))
        print("Saving into:", f"../trained/{run_name}")
        print("===================================================\n")

        if log_path.exists():
            print(f"[INFO] train.log exists; skipping training: {log_path}\n")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            run_train_and_log(cmd, log_path)

        cleanup_epoch_checkpoints(run_dir)

    print("\n[DONE] All grid runs completed (or skipped if already existed).")


if __name__ == "__main__":
    main()

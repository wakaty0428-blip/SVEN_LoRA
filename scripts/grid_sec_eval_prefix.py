# ==============================================================
# grid_sec_eval_prefix.py
#
# Purpose
#   Evaluate the SECURITY RATE of Prefix-Tuning models using sec_eval.py
#   and export summarized results into TXT files.
#
#   This script is the Prefix-Tuning counterpart to:
#
#       grid_sec_eval_orig.py  → evaluates the original base model
#       grid_sec_eval_lora.py  → evaluates LoRA fine-tuned models
#
#   It measures how secure the generated code is after prefix-tuning.
#
# Overview
#   For each selected run and repeated trial, the script performs:
#
#     1) Load the trained prefix model from:
#
#           ../trained/<run_name>/checkpoint-last/
#
#     2) Run sec_eval.py to generate evaluation outputs:
#
#           python sec_eval.py \
#               --model_type prefix \
#               --model_dir <checkpoint-last> \
#               --output_name <run_name>-<trial> \
#               --pretrain_dir <BASE> \
#               --temp 0.4 \
#               --seed <trial_index>
#
#     3) sec_eval.py creates an evaluation folder:
#
#           ../experiments/sec_eval/<output_name>/
#
#     4) Convert the evaluation results into a summarized TXT file:
#
#           python print_results.py --eval_dir <eval_folder> > <output_name>.txt
#
#        Example output file:
#
#           6b-ep5-lr0.01_p8_lm0.180_con41_kl410-1.txt
#
#     5) Delete the intermediate evaluation folder to save disk space.
#
#        Only the TXT summary file is kept.
#
# Skip behavior
#   If the TXT result file already exists and is non-empty,
#   the evaluation for that trial is skipped automatically.
#
# Trial logic
#   Each repeated evaluation uses:
#
#       seed = trial index
#
#   This ensures multiple stochastic generations are evaluated
#   while keeping the experiments reproducible.
#
# Configuration parameters
#
#   BASE
#       Base pretrained model used together with the prefix parameters.
#
#   SELECTED_RUNS
#       List of trained Prefix-Tuning experiment directories to evaluate.
#       These should correspond to folders inside:
#
#           ../trained/<run_name>/
#
#   DEFAULT_REPEATS
#       Number of repeated evaluation runs for each experiment.
#
# Example outputs
#
#       <run_name>-1.txt
#       <run_name>-2.txt
#       ...
#       <run_name>-10.txt
#
# Usage
#
#   Run with default number of trials:
#
#       python grid_sec_eval_prefix.py
#
#   Specify number of trials manually:
#
#       python grid_sec_eval_prefix.py --repeats 10
#
# Output files
#
#   TXT summaries are saved in the current scripts directory:
#
#       scripts/<run_name>-1.txt
#       scripts/<run_name>-2.txt
#       ...
#
#   Intermediate folders under ../experiments/sec_eval/ are deleted
#   automatically after exporting results.
#
# ============================================================== 

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # limit visible GPUs to avoid mismatch errors

import argparse
import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-6B-multi"

# -------------------------------------------------------
# SPECIFY EXACT EXPERIMENTS YOU WANT TO EVALUATE
# -------------------------------------------------------
SELECTED_RUNS = [
    "6b-ep5-lr0.01_p8_lm0.180_con41_kl410",
]

DEFAULT_REPEATS = 10


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_checkpoint(run_name: str, trial_idx: int):
    run_dir = Path("../trained") / run_name
    ckpt_root = run_dir / "checkpoint-last"

    if not ckpt_root.exists():
        print(f"[WARN] {run_name}: missing checkpoint dir: {ckpt_root}")
        return

    # output name + txt name: "<run>-1", "<run>-2", ...
    output_name = f"{run_name}-{trial_idx}"

    # ====================================================
    # 0) SKIP if TXT already exists (and is non-empty)
    # ====================================================
    scripts_dir = Path(".")
    txt_path = scripts_dir / f"{output_name}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
        print(f"[SKIP] TXT already exists (non-empty): {txt_path}")
        return

    seed = trial_idx

    # 1) sec_eval (prefix)
    cmd = (
        f"python sec_eval.py "
        f"--model_type prefix "
        f"--model_dir {ckpt_root} "
        f"--output_name {output_name} "
        f"--pretrain_dir '{BASE}' "
        f"--temp 0.4 "
        f"--seed {seed}"
    )
    run_cmd(cmd)

    eval_folder = Path("../experiments/sec_eval") / output_name
    if not eval_folder.exists():
        print(f"[ERROR] Evaluation folder not found: {eval_folder}")
        return

    # 2) export to txt in scripts/
    cmd = f"python print_results.py --eval_dir {eval_folder} > {txt_path}"
    run_cmd(cmd)
    print(f"[INFO] TXT summary saved to: {txt_path}")

    # 3) delete eval folder
    try:
        shutil.rmtree(eval_folder)
        print(f"[INFO] Deleted evaluation folder: {eval_folder}")
    except Exception as e:
        print(f"[ERROR] Failed to delete folder: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    print("===== EVALUATING SELECTED RUNS (PREFIX) =====")
    print(f"Repeats per run: {args.repeats}")
    for name in SELECTED_RUNS:
        print(" -", name)

    print("\n===== START PREFIX EVALUATION =====")

    for run_name in SELECTED_RUNS:
        print(f"\n=== Evaluating {run_name} ===")
        for trial_idx in range(1, args.repeats + 1):
            print(f"\n[Trial {trial_idx}/{args.repeats}] {run_name} (prefix)")
            evaluate_checkpoint(run_name, trial_idx)

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()

# ==============================================================
# grid_sec_eval_lora.py
#
# Purpose
#   Evaluate the SECURITY RATE of LoRA fine-tuned models using sec_eval.py
#   and export summarized results into TXT files. This script is used to
#   measure how secure the generated code is after LoRA fine-tuning.
#
#   It is the LoRA counterpart to grid_sec_eval_orig.py, which evaluates
#   the original base model.
#
# Overview
#   For each selected training run and control type, the script performs:
#
#     1) Load the LoRA adapter from:
#
#           ../trained/<run_name>/checkpoint-last/<control>
#
#        where <control> is typically:
#
#           sec  → security-aligned adapter
#           vul  → vulnerability-oriented adapter
#
#     2) Run sec_eval.py to generate evaluation outputs:
#
#           python sec_eval.py \
#               --model_type lora \
#               --model_dir <adapter_path> \
#               --output_name <run_name>-<control>-<trial> \
#               --pretrain_dir <BASE> \
#               --temp 0.4 \
#               --seed <trial_index>
#
#     3) sec_eval.py produces evaluation results in:
#
#           ../experiments/sec_eval/<output_name>/
#
#     4) Convert the evaluation results into a summarized TXT file:
#
#           python print_results.py --eval_dir <eval_folder> > <output_name>.txt
#
#        Example output file:
#
#           6b-ep5-lr0.0001_r4_a4_ld0.1_tqkv_proj_wu0_ga2_lm0.180_con41_kl410-sec-1.txt
#
#     5) Delete the intermediate evaluation folder to save disk space.
#
# Skip behavior
#   If the TXT result file already exists and is non-empty, the evaluation
#   for that trial is skipped automatically.
#
# Trial logic
#   Each repeated evaluation uses:
#
#       seed = trial index
#
#   This ensures that multiple stochastic generations are evaluated
#   while keeping the experiment reproducible.
#
# Configuration parameters
#
#   BASE
#       Base pretrained model used when loading LoRA adapters.
#
#   SELECTED_RUNS
#       List of trained LoRA experiment directories to evaluate.
#       These should correspond to folders inside:
#
#           ../trained/<run_name>/
#
#   CONTROLS
#       Specifies which adapters to evaluate:
#
#           sec → security-aligned generation
#           vul → vulnerability-oriented generation
#
#   NUM_REPEATS
#       Number of repeated evaluation runs for each (run_name, control).
#
# Example outputs
#
#       <run_name>-sec-1.txt
#       <run_name>-sec-2.txt
#       ...
#       <run_name>-sec-10.txt
#
# Usage
#
#   Run with default number of repeats:
#
#       python grid_sec_eval_lora.py
#
#   Specify number of repeats manually:
#
#       python grid_sec_eval_lora.py --repeats 10
#
# Output files
#
#   TXT summaries are saved in the current scripts directory:
#
#       scripts/<run_name>-sec-1.txt
#       scripts/<run_name>-sec-2.txt
#       ...
#
#   Intermediate folders under ../experiments/sec_eval/ are deleted
#   automatically after exporting results.
#
# ============================================================== 


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"    # limit visible GPUs to avoid mismatch errors

import argparse
import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-6B-multi"  # change depending on base model

# -------------------------------------------------------
# SPECIFY EXACT EXPERIMENTS YOU WANT TO EVALUATE
# -------------------------------------------------------
SELECTED_RUNS = [
    "6b-ep5-lr0.0001_r4_a4_ld0.1_tqkv_proj_wu0_ga2_lm0.180_con41_kl410",
]

# CONTROLS = ["sec", "vul"]  # evaluate both in this order
CONTROLS = ["sec"]
NUM_REPEATS = 10           # run each (run_name, control) this many times


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_one(run_name: str, control: str, trial_idx: int):
    # Example: "<run_name>-sec-1", "<run_name>-sec-2", ...
    output_name = f"{run_name}-{control}-{trial_idx}"

    # ====================================================
    # 0) SKIP if TXT already exists (and is non-empty)
    # ====================================================
    scripts_dir = Path(".")  # run from sven_old/scripts
    txt_path = scripts_dir / f"{output_name}.txt"

    if txt_path.exists() and txt_path.stat().st_size > 0:
        print(f"[SKIP] TXT already exists (non-empty): {txt_path}")
        return

    run_dir = Path("../trained") / run_name
    ckpt = run_dir / "checkpoint-last"
    adapter_path = ckpt / control

    if not adapter_path.exists():
        print(f"[WARN] {run_name}: missing checkpoint or adapter <{control}> at {adapter_path}")
        return

    # ====================================================
    # 1) RUN sec_eval.py → produces evaluation folder
    # ====================================================
    # Fair + simple: seed 1 for repeat 1, seed 2 for repeat 2, ...
    seed = trial_idx

    cmd = (
        f"python sec_eval.py "
        f"--model_type lora "
        f"--model_dir {adapter_path} "
        f"--output_name {output_name} "
        f"--pretrain_dir '{BASE}' "
        f"--temp 0.4 "
        f"--seed {seed}"
    )
    run_cmd(cmd)

    # sec_eval.py produces: ../experiments/sec_eval/<output_name>/
    eval_folder = Path("../experiments/sec_eval") / output_name
    if not eval_folder.exists():
        print(f"[ERROR] Evaluation folder not found: {eval_folder}")
        return

    # ====================================================
    # 2) EXPORT TO TXT inside scripts/ directory
    # ====================================================
    cmd = f"python print_results.py --eval_dir {eval_folder} > {txt_path}"
    run_cmd(cmd)
    print(f"[INFO] TXT summary saved to: {txt_path}")

    # ====================================================
    # 3) DELETE evaluation folder to save space
    # ====================================================
    try:
        shutil.rmtree(eval_folder)
        print(f"[INFO] Deleted evaluation folder: {eval_folder}")
    except Exception as e:
        print(f"[ERROR] Failed to delete folder: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=NUM_REPEATS, help="Number of repeated eval runs per experiment")
    args = parser.parse_args()

    print("===== EVALUATING SELECTED RUNS (AUTO: sec -> vul) =====")
    print(f"Repeats per (run, control): {args.repeats}")
    for name in SELECTED_RUNS:
        print(" -", name)

    for run_name in SELECTED_RUNS:
        print(f"\n=== Evaluating {run_name} ===")
        for control in CONTROLS:
            print(f"\n--- Control: {control} ---")
            for trial_idx in range(1, args.repeats + 1):
                print(f"\n[Trial {trial_idx}/{args.repeats}] {run_name} ({control})")
                evaluate_one(run_name, control, trial_idx)

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()

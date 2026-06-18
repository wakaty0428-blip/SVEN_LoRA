import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limit visible GPUs to avoid mismatch errors

import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-6B-multi"

# =======================================================
# HumanEval generation config
# =======================================================
MAX_GEN_LEN = 200
NUM_SAMPLES_PER_GEN = 10

# =======================================================
# Trial config (trial i -> seed i)
# =======================================================
NUM_TRIALS = 10  # trial 1->seed 1, trial 2->seed 2, ...


# -------------------------------------------------------
# SPECIFY EXACT EXPERIMENTS YOU WANT TO RUN HUMAN EVAL ON
# (PREFIX runs: checkpoint-last contains both sec/vul prefixes)
# -------------------------------------------------------
SELECTED_RUNS = [
    "6b-lr0.01_p16_lm0.180_con41_kl410",
]

# Run sec then vul automatically
# CONTROLS = ["sec", "vul"]
CONTROLS = ["sec"]


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_human_eval_prefix(run_name: str, control: str, trial_idx: int):
    """
    trial_idx is 1-indexed.
      trial 1 -> seed 1
      trial 2 -> seed 2
      ...
    """
    seed = trial_idx

    run_dir = Path("../trained") / run_name
    ckpt_root = run_dir / "checkpoint-last"

    if not ckpt_root.exists():
        print(f"[WARN] {run_name}: missing checkpoint dir: {ckpt_root}")
        return

    # Output name includes trial + seed so every run is unique
    output_name = f"{run_name}-{control}-trial{trial_idx}-seed{seed}"

    # ====================================================
    # SKIP IF TXT ALREADY EXISTS
    # ====================================================
    scripts_dir = Path(".")
    txt_path = scripts_dir / f"human-eval-{output_name}.txt"

    if txt_path.exists():
        print(f"[SKIP] TXT already exists → {txt_path.name}")
        return
    
    # ============================================
    # (1) RUN human_eval_gen.py (PREFIX)
    #    NOTE: prefix uses checkpoint-last root
    # ============================================
    gen_cmd = (
        f"python human_eval_gen.py "
        f"--model_type prefix "
        f"--model_dir {ckpt_root} "
        f"--control {control} "
        f"--pretrain_dir '{BASE}' "
        f"--output_name {output_name} "
        f"--max_gen_len {MAX_GEN_LEN} "
        f"--num_samples_per_gen {NUM_SAMPLES_PER_GEN} "
        f"--seed {seed}"
    )
    run_cmd(gen_cmd)

    # ============================================
    # (2) RUN human_eval_exec.py
    # ============================================
    exec_cmd = f"python human_eval_exec.py --output_name {output_name}"
    run_cmd(exec_cmd)

    # ============================================
    # (3) EXPORT TXT USING print_results.py
    # ============================================
    eval_folder = Path("../experiments/human_eval") / output_name
    if not eval_folder.exists():
        print(f"[ERROR] Evaluation folder not found: {eval_folder}")
        return

    scripts_dir = Path(".")  # run from scripts/
    txt_path = scripts_dir / f"human-eval-{output_name}.txt"

    txt_cmd = (
        f"python print_results.py "
        f"--eval_type human_eval "
        f"--eval_dir {eval_folder} "
        f"> {txt_path}"
    )
    run_cmd(txt_cmd)
    print(f"[INFO] TXT summary saved to: {txt_path}")

    # ============================================
    # (4) DELETE ORIGINAL HEAVY HUMAN_EVAL FOLDER
    # ============================================
    try:
        shutil.rmtree(eval_folder)
        print(f"[INFO] Deleted heavy folder: {eval_folder}")
    except Exception as e:
        print(f"[ERROR] Could not delete folder: {e}")


def main():
    print("===== HUMAN EVAL ON SELECTED RUNS (PREFIX, MULTI-TRIAL) =====")
    for name in SELECTED_RUNS:
        print(" -", name)

    print(f"\n[CONFIG] max_gen_len={MAX_GEN_LEN}, num_samples_per_gen={NUM_SAMPLES_PER_GEN}")
    print(f"[CONFIG] NUM_TRIALS={NUM_TRIALS} (trial i -> seed i)")

    for run_name in SELECTED_RUNS:
        print(f"\n=== Running HumanEval for {run_name} ===")

        for control in CONTROLS:
            for trial_idx in range(1, NUM_TRIALS +1):
                seed = trial_idx

                print(f"\n--- [{run_name}] START {control.upper()} trial={trial_idx} seed={seed} ---")
                evaluate_human_eval_prefix(run_name, control, trial_idx)
                print(f"--- [{run_name}] DONE {control.upper()} trial={trial_idx} ---")

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()

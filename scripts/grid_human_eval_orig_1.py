import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limit visible GPUs

import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-2B-multi"

# =======================================================
# HumanEval generation config
# =======================================================
MAX_GEN_LEN = 150
NUM_SAMPLES_PER_GEN = 25

# =======================================================
# Trial config (trial i -> seed i)
# =======================================================
NUM_TRIALS = 10  # trial 1->seed 1, trial 2->seed 2, ...

# -------------------------------------------------------
# "RUNS" here are just labels for output naming.
# For original LM, there is no ../trained/<run>.
# -------------------------------------------------------
SELECTED_TAGS = [
    "2b-lm",  # output label only
]

# If your pipeline uses different prompt sets via --control, keep it.
# Otherwise you can set CONTROLS = ["sec"] only or even remove control usage.
CONTROLS = ["sec"]  # or ["sec", "vul"]


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_human_eval_lm(tag: str, control: str, trial_idx: int):
    """
    Base/original LM evaluation (no checkpoint).
    trial_idx is 1-indexed.
      trial 1 -> seed 1
      trial 2 -> seed 2
      ...
    """
    seed = trial_idx

    # Output name includes trial + seed so every run is unique
    output_name = f"{tag}-{control}-trial{trial_idx}-seed{seed}"

    # ====================================================
    # SKIP IF TXT ALREADY EXISTS
    # ====================================================
    scripts_dir = Path(".")  # run from scripts/
    txt_path = scripts_dir / f"human-eval-{output_name}.txt"
    if txt_path.exists():
        print(f"[SKIP] TXT already exists → {txt_path.name}")
        return

    # ============================================
    # (1) RUN human_eval_gen.py (LM / original)
    # ============================================
    gen_cmd = (
        f"python human_eval_gen.py "
        f"--model_type lm "
        f"--pretrain_dir '{BASE}' "
        f"--control {control} "
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
    print("===== HUMAN EVAL ON BASE MODEL (LM, MULTI-TRIAL) =====")
    for tag in SELECTED_TAGS:
        print(" -", tag)

    print(f"\n[CONFIG] BASE(pretrain_dir)={BASE}")
    print(f"[CONFIG] max_gen_len={MAX_GEN_LEN}, num_samples_per_gen={NUM_SAMPLES_PER_GEN}")
    print(f"[CONFIG] NUM_TRIALS={NUM_TRIALS} (trial i -> seed i)")

    for tag in SELECTED_TAGS:
        print(f"\n=== Running HumanEval for tag={tag} (base LM) ===")

        for control in CONTROLS:
            for trial_idx in range(1, NUM_TRIALS -6):
                seed = trial_idx
                print(f"\n--- [{tag}] START {control.upper()} trial={trial_idx} seed={seed} ---")
                evaluate_human_eval_lm(tag, control, trial_idx)
                print(f"--- [{tag}] DONE {control.upper()} trial={trial_idx} ---")

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()
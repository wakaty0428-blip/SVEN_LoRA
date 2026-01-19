import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # this is for limiting the cuda that prevent from seeing different cuda that will cause error
import argparse
import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-350M-multi"

# -------------------------------------------------------
# SPECIFY EXACT EXPERIMENTS YOU WANT TO EVALUATE
# -------------------------------------------------------
SELECTED_RUNS = [
    "350m-lr0.01_p16_lm0.200_con41_kl390",
    "350m-lr0.01_p16_lm0.220_con39_kl390",
    "350m-lr0.01_p16_lm0.220_con41_kl370",
    "350m-lr0.01_p16_lm0.240_con35_kl410",
    "350m-lr0.01_p16_lm0.260_con39_kl350",
    "350m-lr0.01_p16_lm0.260_con41_kl330",
    "350m-lr0.01_p16_lm0.280_con31_kl410",
    "350m-lr0.01_p16_lm0.280_con41_kl310",
    "350m-lr0.01_p16_lm0.300_con29_kl410",
    "350m-lr0.01_p16_lm0.320_con29_kl390",
    "350m-lr0.01_p16_lm0.320_con33_kl350",
    "350m-lr0.01_p16_lm0.340_con33_kl330",
    "350m-lr0.01_p16_lm0.360_con25_kl390",
    "350m-lr0.01_p16_lm0.360_con27_kl370",
    "350m-lr0.01_p16_lm0.360_con37_kl270",
    "350m-lr0.01_p16_lm0.400_con31_kl290",
    "350m-lr0.01_p16_lm0.400_con35_kl250",
    "350m-lr0.01_p16_lm0.440_con27_kl290",
    "350m-lr0.01_p16_lm0.500_con25_kl250",
]


def run_cmd(cmd):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_checkpoint(run_name):
    run_dir = Path("../trained") / run_name
    ckpt_root = run_dir / "checkpoint-last"

    if not ckpt_root.exists():
        print(f"[WARN] {run_name}: missing checkpoint dir: {ckpt_root}")
        return

    # Prefix evaluation output folder name (single folder contains both sec/vul results)
    output_name = run_name

    # ====================================================
    # 1. RUN sec_eval.py (PREFIX)
    #    - No --control (sec/vul already inside)
    # ====================================================
    cmd = (
        f"python sec_eval.py "
        f"--model_type prefix "
        f"--model_dir {ckpt_root} "
        f"--output_name {output_name} "
        f"--pretrain_dir '{BASE}' "
        f"--temp 0.4"
    )
    run_cmd(cmd)

    # sec_eval.py produces:
    # ../experiments/sec_eval/<output_name>/
    eval_folder = Path("../experiments/sec_eval") / output_name
    if not eval_folder.exists():
        print(f"[ERROR] Evaluation folder not found: {eval_folder}")
        return

    # ====================================================
    # 2. EXPORT TO TXT INSIDE scripts/ DIRECTORY
    # ====================================================
    scripts_dir = Path(".")  # assume you run from sven_old/scripts
    txt_path = scripts_dir / f"{output_name}.txt"

    cmd = (
        f"python print_results.py "
        f"--eval_dir {eval_folder} "
        f"> {txt_path}"
    )
    run_cmd(cmd)

    print(f"[INFO] TXT summary saved to: {txt_path}")

    # ====================================================
    # 3. DELETE ORIGINAL EVALUATION FOLDER TO SAVE SPACE
    # ====================================================
    try:
        shutil.rmtree(eval_folder)
        print(f"[INFO] Deleted evaluation folder: {eval_folder}")
    except Exception as e:
        print(f"[ERROR] Failed to delete folder: {e}")


def main():
    # keep argparse for symmetry / future options, but no control needed
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("===== EVALUATING SELECTED RUNS (PREFIX) =====")
    for name in SELECTED_RUNS:
        print(" -", name)

    print("\n===== START PREFIX EVALUATION =====")

    for run_name in SELECTED_RUNS:
        print(f"\n=== Evaluating {run_name} ===")
        evaluate_checkpoint(run_name)

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()

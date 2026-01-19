import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # this is for limiting the cuda that prevent from seeing different cuda that will cause error
import argparse
import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-2B-multi"  # change depending on base model

# -------------------------------------------------------
# SPECIFY EXACT EXPERIMENTS YOU WANT TO EVALUATE
# -------------------------------------------------------
SELECTED_RUNS = [
    "2b-lr0.0001_r8_lm0.180_con41_kl410",
    "2b-lr0.0001_r8_lm0.200_con41_kl390",
    "2b-lr0.0001_r8_lm0.220_con39_kl390",
    "2b-lr0.0001_r8_lm0.260_con39_kl350",
    "2b-lr0.0001_r8_lm0.280_con31_kl410",
    "2b-lr0.0001_r8_lm0.320_con29_kl390",
    "2b-lr0.0001_r8_lm0.320_con33_kl350",
    "2b-lr0.0001_r8_lm0.340_con33_kl330",
    "2b-lr0.0001_r8_lm0.360_con25_kl390",
    "2b-lr0.0001_r8_lm0.360_con27_kl370",
    "2b-lr0.0001_r8_lm0.360_con37_kl270",
    "2b-lr0.0001_r8_lm0.400_con35_kl250",
    "2b-lr0.0001_r8_lm0.440_con27_kl290",
]

CONTROLS = ["sec", "vul"]  # automatically evaluate both, in this order


def run_cmd(cmd):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_one(run_name: str, control: str):
    run_dir = Path("../trained") / run_name
    ckpt = run_dir / "checkpoint-last"
    adapter_path = ckpt / control

    if not adapter_path.exists():
        print(f"[WARN] {run_name}: missing checkpoint or adapter <{control}> at {adapter_path}")
        return

    output_name = f"{run_name}-{control}"

    # ====================================================
    # 1) RUN sec_eval.py → produces evaluation folder
    # ====================================================
    cmd = (
        f"python sec_eval.py "
        f"--model_type lora "
        f"--model_dir {adapter_path} "
        f"--output_name {output_name} "
        f"--pretrain_dir '{BASE}' "
        f"--temp 0.4"
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
    scripts_dir = Path(".")  # run from sven_old/scripts
    txt_path = scripts_dir / f"{output_name}.txt"

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
    # No --control needed anymore; keep argparse in case you add options later
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("===== EVALUATING SELECTED RUNS (AUTO: sec -> vul) =====")
    for name in SELECTED_RUNS:
        print(" -", name)

    for run_name in SELECTED_RUNS:
        print(f"\n=== Evaluating {run_name} ===")
        for control in CONTROLS:
            print(f"\n--- Control: {control} ---")
            evaluate_one(run_name, control)

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()

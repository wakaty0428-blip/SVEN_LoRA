import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # this is for limiting the cuda that prevent from seeing different cuda that will cause error
import argparse
import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-350M-multi"

# -------------------------------------------------------
# SPECIFY EXACT EXPERIMENTS YOU WANT TO RUN HUMAN EVAL ON
# -------------------------------------------------------
SELECTED_RUNS = [
    "350m-lr0.0001_r8_lm0.180_con41_kl410",
    "350m-lr0.0001_r8_lm0.200_con41_kl390",
    "350m-lr0.0001_r8_lm0.220_con39_kl390",
    "350m-lr0.0001_r8_lm0.220_con41_kl370",
    "350m-lr0.0001_r8_lm0.240_con35_kl410",
    "350m-lr0.0001_r8_lm0.260_con39_kl350",
    "350m-lr0.0001_r8_lm0.260_con41_kl330",
    "350m-lr0.0001_r8_lm0.280_con31_kl410",
    "350m-lr0.0001_r8_lm0.280_con41_kl310",
    "350m-lr0.0001_r8_lm0.300_con29_kl410",
    "350m-lr0.0001_r8_lm0.320_con29_kl390",
    "350m-lr0.0001_r8_lm0.320_con33_kl350",
    "350m-lr0.0001_r8_lm0.340_con33_kl330",
    "350m-lr0.0001_r8_lm0.360_con25_kl390",
    "350m-lr0.0001_r8_lm0.360_con27_kl370",
    "350m-lr0.0001_r8_lm0.360_con37_kl270",
    "350m-lr0.0001_r8_lm0.400_con31_kl290",
    "350m-lr0.0001_r8_lm0.440_con27_kl290",
    "350m-lr0.0001_r8_lm0.500_con25_kl250",
]

# Order matters: run sec first, then vul
CONTROLS = ["sec", "vul"]


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_human_eval(run_name: str, control: str):
    run_dir = Path("../trained") / run_name
    ckpt = run_dir / "checkpoint-last"
    adapter_path = ckpt / control  # checkpoint-last/sec or checkpoint-last/vul

    if not adapter_path.exists():
        print(f"[WARN] {run_name}: missing adapter folder <{control}> at {adapter_path}")
        return

    # Output name should match your sec_eval naming style: "<run>-sec" / "<run>-vul"
    output_name = f"{run_name}-{control}"

    # ============================================
    # (1) RUN human_eval_gen.py
    # ============================================
    gen_cmd = (
        f"python human_eval_gen.py "
        f"--model_type lora "
        f"--model_dir {adapter_path} "
        f"--control {control} "
        f"--pretrain_dir '{BASE}' "
        f"--output_name {output_name}"
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

    scripts_dir = Path(".")  # current directory = scripts/
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
    print("===== HUMAN EVAL ON SELECTED RUNS =====")
    for name in SELECTED_RUNS:
        print(" -", name)

    for run_name in SELECTED_RUNS:
        print(f"\n=== Running HumanEval for {run_name} ===")

        for control in CONTROLS:
            print(f"\n--- [{run_name}] START {control.upper()} ---")
            evaluate_human_eval(run_name, control)
            print(f"--- [{run_name}] DONE {control.upper()} ---")

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import argparse
import subprocess
import shutil
from pathlib import Path

BASE = "Salesforce/codegen-350M-multi"

SELECTED_TAGS = [
    "350m-lm",
]

DEFAULT_REPEATS = 10


def run_cmd(cmd: str):
    print(f"\n[RUN] {cmd}\n")
    subprocess.run(cmd, shell=True, check=True)


def evaluate_lm(tag: str, trial_idx: int):
    output_name = f"{tag}-{trial_idx}"

    scripts_dir = Path(".")
    txt_path = scripts_dir / f"{output_name}.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
        print(f"[SKIP] TXT already exists (non-empty): {txt_path}")
        return

    seed = trial_idx

    # 1) sec_eval (LM / original)  ✅ force model_dir to 6B
    cmd = (
        f"python sec_eval.py "
        f"--model_type lm "
        f"--model_dir '{BASE}' "
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

    # 2) export to txt
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

    print("===== EVALUATING SELECTED TAGS (LM / ORIGINAL) =====")
    print(f"BASE: {BASE}")
    print(f"Repeats per tag: {args.repeats}")
    for tag in SELECTED_TAGS:
        print(" -", tag)

    for tag in SELECTED_TAGS:
        print(f"\n=== Evaluating {tag} ===")
        for trial_idx in range(5, args.repeats + 1):
            print(f"\n[Trial {trial_idx}/{args.repeats}] {tag} (lm)")
            evaluate_lm(tag, trial_idx)

    print("\n===== ALL DONE =====")


if __name__ == "__main__":
    main()
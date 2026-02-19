#!/usr/bin/env python3
"""
kfold_crossval.py

K-fold cross validation runner for SVEN without modifying existing code.

Fix:
- train.py does NOT accept an extra '--' separator.
- So we strip a leading '--' from passthrough args if present.

python kfold_crossval.py \
  --data_dir ../data_train_val \
  --k 10 \
  --seed 42 \
  --fold_root ../data_kfold_tmp \
  --output_dir ../trained \
  --run_prefix lora-qkvout \
  -- \
  --model_type lora \
  --pretrain_dir Salesforce/codegen-350M-multi \
  --learning_rate 0.0001 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --lora_target_modules qkv_proj,out_proj \
  --contrastive_loss_ratio 30 \
  --kl_loss_ratio 400 \
  --lm_loss_ratio 0.3 \
  --num_train_epochs 7

"""

import argparse
import json
import random
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


VAL_LOSS_RE = re.compile(r"val epoch\s+\d+:[^\n]*,\s*loss:\s*([0-9]*\.?[0-9]+)")


def read_jsonl_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(errors="ignore").splitlines(True)  # keep newline


def write_jsonl_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line)


def discover_cwe_files(data_dir: Path) -> List[str]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    files = []
    if train_dir.exists():
        files = sorted([p.name for p in train_dir.glob("*.jsonl")])
    if not files and val_dir.exists():
        files = sorted([p.name for p in val_dir.glob("*.jsonl")])

    if not files:
        raise FileNotFoundError(f"No *.jsonl found under {train_dir} or {val_dir}")
    return files


def build_pool_per_cwe(data_dir: Path, cwe_file: str) -> List[str]:
    train_path = data_dir / "train" / cwe_file
    val_path = data_dir / "val" / cwe_file
    lines = read_jsonl_lines(train_path) + read_jsonl_lines(val_path)

    # quick sanity: ensure parseable json for a few lines
    for ln in lines[:3]:
        json.loads(ln)

    return lines


def kfold_indices(n: int, k: int, rng: random.Random) -> List[List[int]]:
    idx = list(range(n))
    rng.shuffle(idx)
    folds = [[] for _ in range(k)]
    for i, j in enumerate(idx):
        folds[i % k].append(j)
    return folds


def make_fold_dataset(
    fold_root: Path,
    fold_id: int,
    k: int,
    data_dir: Path,
    cwe_files: List[str],
    seed: int,
) -> Path:
    rng = random.Random(seed)

    fold_dir = fold_root / f"fold_{fold_id}"
    train_out = fold_dir / "train"
    val_out = fold_dir / "val"

    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    for cwe_file in cwe_files:
        pool = build_pool_per_cwe(data_dir, cwe_file)

        if len(pool) < k:
            write_jsonl_lines(train_out / cwe_file, pool)
            write_jsonl_lines(val_out / cwe_file, [])
            continue

        folds = kfold_indices(len(pool), k, rng)
        val_idx = set(folds[fold_id])

        train_lines = [pool[i] for i in range(len(pool)) if i not in val_idx]
        val_lines = [pool[i] for i in range(len(pool)) if i in val_idx]

        write_jsonl_lines(train_out / cwe_file, train_lines)
        write_jsonl_lines(val_out / cwe_file, val_lines)

    return fold_dir


def extract_last_val_loss(train_log: Path) -> float:
    text = train_log.read_text(errors="ignore")
    vals = VAL_LOSS_RE.findall(text)
    if not vals:
        raise ValueError(f"No val loss found in {train_log}")
    return float(vals[-1])


def run_train(
    train_py: Path,
    fold_data_dir: Path,
    output_name: str,
    extra_train_args: List[str],
    log_path: Path,
) -> None:
    cmd = [
        "python",
        str(train_py),
        "--output_name",
        output_name,
        "--data_dir",
        str(fold_data_dir),
    ] + extra_train_args

    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n===================================================")
    print(f"▶ Training: {output_name}")
    print("data_dir:", fold_data_dir)
    print("Command:", " ".join(cmd))
    print("===================================================\n")

    with open(log_path, "w") as f:
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")
            f.write(line)
        p.wait()
        if p.returncode != 0:
            raise subprocess.CalledProcessError(p.returncode, cmd)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="../data_train_val")
    ap.add_argument("--fold_root", type=str, default="../data_kfold_tmp")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_py", type=str, default="train.py")
    ap.add_argument("--output_dir", type=str, default="../trained")
    ap.add_argument("--run_prefix", type=str, default="kfold")
    ap.add_argument("--skip_if_exists", action="store_true")

    # collect passthrough args
    args, passthrough = ap.parse_known_args()

    # ---- FIX: strip a leading "--" if user writes " -- --model_type ..."
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    data_dir = Path(args.data_dir).resolve()
    fold_root = Path(args.fold_root).resolve()
    train_py = Path(args.train_py).resolve()
    output_dir = Path(args.output_dir).resolve()

    cwe_files = discover_cwe_files(data_dir)
    print("[INFO] Found CWE files:", len(cwe_files))
    print("       Example:", cwe_files[:5])

    results: List[Tuple[int, float, Path]] = []

    for fold_id in range(args.k):
        fold_data_dir = make_fold_dataset(
            fold_root=fold_root,
            fold_id=fold_id,
            k=args.k,
            data_dir=data_dir,
            cwe_files=cwe_files,
            seed=args.seed,
        )

        output_name = f"{args.run_prefix}-fold{fold_id}-k{args.k}-seed{args.seed}"
        run_dir = output_dir / output_name
        train_log = run_dir / "train.log"

        extra_train_args = ["--output_dir", str(output_dir)] + passthrough

        if args.skip_if_exists and train_log.exists():
            print(f"[SKIP] train.log exists: {train_log}")
        else:
            run_train(
                train_py=train_py,
                fold_data_dir=fold_data_dir,
                output_name=output_name,
                extra_train_args=extra_train_args,
                log_path=train_log,
            )

        val_loss = extract_last_val_loss(train_log)
        results.append((fold_id, val_loss, train_log))

        print(f"\n[FOLD RESULT] fold={fold_id} final_val_loss={val_loss:.6f}")
        print(f"  train.log: {train_log}")

    # Summary
    results_sorted = sorted(results, key=lambda x: x[1])
    print("\n==================== K-FOLD SUMMARY ====================")
    for fold_id, val_loss, logp in results_sorted:
        print(f"fold={fold_id}  val_loss={val_loss:.6f}  log={logp}")
    avg = sum(x[1] for x in results) / len(results)
    print(f"Average val_loss over {args.k} folds: {avg:.6f}")
    print("========================================================\n")


if __name__ == "__main__":
    main()

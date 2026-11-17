#!/usr/bin/env python3
# ==============================================================
# grid_train_lora_new.py
#
# Purpose
#   Search the weights of the multi-objective loss
#   (LM loss, contrastive loss, KL loss) for LoRA on CodeGen.
#   Method-specific hyperparameters (learning rate, rank, target modules, ...) are
#   kept fixed, and ONLY the loss weights are changed.
#
# --------------------------------------------------------------
# How the weights are represented (they always sum to 1)
# --------------------------------------------------------------
#   Value passed to train.py  ->  actual weight
#       lm  : lm_loss_ratio          (float) -> used as is
#       con : contrastive_loss_ratio (int)   -> con / 100
#       kl  : kl_loss_ratio          (int)   -> kl  / 1000
#   Every candidate satisfies:  lm + con/100 + kl/1000 = 1
#   (checked by validate_weights(); the script stops if it is violated).
#
# --------------------------------------------------------------
# How the candidates are chosen
# --------------------------------------------------------------
#   - Candidates are defined as RATIOS first, then normalized so that the
#     weights sum to 1.
#     Reason: if we vary the weights directly under the sum-to-1 constraint
#     (e.g. in steps of 0.1), raising one weight automatically lowers the
#     others, so we cannot tell whether an effect comes from "a stronger
#     contrastive loss" or from "a weaker LM loss".
#   - The center is the ratio of the original SVEN paper:
#         LM : contrastive : KL = 1 : 4 : 1.6
#     (= 0.152 : 0.606 : 0.242 when normalized to sum to 1).
#   - The contrastive and KL ratios are each multiplied by
#     0.5, 1, 2  ->  3 x 3 = 9 candidates.
#   - After normalization, con is rounded to steps of 0.01 and kl to steps
#     of 0.001 (the integer format of train.py), and the remainder is given
#     to lm so that the sum is exactly 1.
#   - The previous setting (REFERENCE_WEIGHTS) is also trained for comparison.
#   - LoRA was the method whose functional correctness dropped the most with
#     the multi-loss setting, so a key check is whether the candidates with
#     a larger KL ratio (x2) recover functionality.
#
# --------------------------------------------------------------
# Selection criterion (which weights to adopt)
# --------------------------------------------------------------
#   - Do NOT select by the weighted total validation loss.
#     Changing the weights changes the scale of the metric itself, so the
#     totals of different candidates are not comparable. Because the LM loss
#     is much larger in absolute value, a higher LM weight always looks
#     better and the search sticks to the upper bound of the LM weight.
#   - Do NOT select by security rate alone either. It tends to pick the
#     setting with the strongest contrastive loss, which may produce code
#     that does not work.
#   - Criterion (SELECTION_CRITERION):
#       Among the candidates whose functional correctness on the
#       VALIDATION prompts drops by at most epsilon (e.g. 1 point) from the
#       no-fine-tuning baseline, choose the one with the highest
#       security rate.
#   - Keep the validation prompts separate from the final test prompts.
#     Selecting weights on the test set makes the results optimistic.
#
# --------------------------------------------------------------
# Overall workflow
# --------------------------------------------------------------
#   Stage 1 (STAGE = "screen", this script)
#     Train the 9 candidates + the previous setting with a short schedule
#     (SCREEN_EPOCHS). Compare each loss term on the validation set
#     WITHOUT the weights applied, and drop clearly bad candidates
#     (screening with proxy metrics):
#       - LM loss or KL loss clearly worse -> functionality likely degraded
#       - contrastive loss does not go down -> security is not improving
#     Note: if train.py logs weighted values, divide them by the weight
#           before comparing.
#     Note: first check on 2-3 candidates that the proxy metrics move
#           together with the real metrics.
#
#   Stage 2 (evaluation script, outside this script)
#     Evaluate the remaining candidates with a light evaluation
#     (stratified subset of the validation prompts, one temperature,
#     pass@1) and pick the top 2 by the selection criterion above.
#
#   Stage 3 (STAGE = "final", this script)
#     Write the top 2 candidates into FINAL_WEIGHTS as (lm, con, kl), and
#     train them with the normal schedule (FINAL_EPOCHS) and several seeds
#     (FINAL_SEEDS). Run the full evaluation and make the final decision,
#     checking that the difference is larger than the seed-to-seed variance.
#
#   To make the search even cheaper, use SEARCH_MODE (one axis at a time):
#     "coordinate_con" : fix KL at x1 and vary only contrastive (3 runs)
#     "coordinate_kl"  : fix contrastive at BEST_CON_MULTIPLIER and vary KL
#                        (runs already trained are skipped automatically)
#
# --------------------------------------------------------------
# Other behavior
# --------------------------------------------------------------
#   - LoRA hyperparameters (LEARNING_RATES, ...) can be lists, but keep ONE
#     value each while searching the loss weights.
#   - lora_alpha = 4 * r (set in build_runs()).
#   - Training stdout is written to BOTH the terminal and
#     ../trained/<run_name>/train.log.
#   - A run is skipped if checkpoint-epoch-* or checkpoint-last exists.
#     A run that only has train.log (crashed run) is trained again.
#   - If a run fails, the remaining runs continue; failed runs are listed
#     at the end.
#   - After each run, intermediate epoch checkpoints are deleted. Only the
#     final epoch checkpoint, checkpoint-last and train.log are kept.
#   - run_config.json is written into each run directory (stage, actual
#     weights, seed, selection criterion, ...) so that evaluation /
#     aggregation scripts can read it.
#
# Run naming convention
#   <base>-ep<ep>-lr<lr>_r<r>_a<alpha>_ld<dropout>_t<target>_wu<warmup>_ga<ga>
#     _lm<lm>_con<con>_kl<kl>[_s<seed>]
# ==============================================================

import os
import re
import json
import argparse
import itertools
import subprocess
from pathlib import Path
from typing import List, Dict

# ==============================================================
# Run settings
# ==============================================================
STAGE = "screen"        # "screen" (Stage 1) or "final" (Stage 3)
DRY_RUN = False         # True: only print candidates and commands, no training
SKIP_EXISTING = True    # True: skip runs that are already trained

# ==============================================================
# Common settings
# ==============================================================
PRETRAIN = "Salesforce/codegen-350M-multi"
MODEL_TYPE = "lora"
BASE = "350m"
TRAINED_ROOT = Path("../trained")

# ==============================================================
# LoRA hyperparameters (keep ONE value each while searching the weights)
# ==============================================================
LEARNING_RATES = [1e-04]
LORA_TARGETS = ["qkv_proj"]          # e.g. ["out_proj", "qkv_proj,out_proj"]
LORA_RS = [8]
LORA_DROPOUTS = [0.1]
WARMUP_STEPS_LIST = [0]              # e.g. [0, 200]
# If --grad_acc_steps is given on the CLI, it overrides this list
GRAD_ACC_STEPS_LIST = [2]

# ==============================================================
# Loss-weight search settings
# ==============================================================
# Ratio of the original SVEN paper (LM = 1)
SVEN_RATIO = {"lm": 1.0, "con": 4.0, "kl": 1.6}

# "grid" | "coordinate_con" | "coordinate_kl" (see header)
SEARCH_MODE = "grid"
CON_MULTIPLIERS = [0.5, 1.0, 2.0]   # multipliers for the contrastive ratio
KL_MULTIPLIERS  = [0.5, 1.0, 2.0]   # multipliers for the KL ratio
BEST_CON_MULTIPLIER = 1.0           # contrastive multiplier fixed in "coordinate_kl"

# Previous setting for comparison: (lm, con, kl). Must sum to 1.
#   Old script: FIXED_CONTRASTIVE_RATIO = 41 (-> 0.41),
#   FIXED_KL_RATIO = 410 (-> 0.410), lm = 1 - 0.41 - 0.41 = 0.18
REFERENCE_WEIGHTS = [(0.180, 41, 410)]

# Candidates for Stage 3: (lm, con, kl). Fill in after Stage 2.
# e.g. FINAL_WEIGHTS = [(0.148, 61, 242), (0.120, 49, 390)]
FINAL_WEIGHTS = []

# Selection criterion (recorded in run_config.json)
SELECTION_CRITERION = (
    "Among candidates whose functional correctness on the validation prompts "
    "drops by at most epsilon (e.g. 1 point) from the no-fine-tuning baseline, "
    "choose the one with the highest security rate. "
    "Weighted validation loss is NOT used for selection."
)

# ==============================================================
# Epochs and seeds
# ==============================================================
SCREEN_EPOCHS = 3          # Stage 1: short training (enough to rank candidates)
FINAL_EPOCHS  = 7          # Stage 3: normal training
SCREEN_SEEDS  = [1]
FINAL_SEEDS   = [1]
PASS_SEED     = True       # set to False if train.py has no --seed option


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


# ---------- Weight generation and validation ----------
def weights_sum(lm, con, kl):
    return lm + con / 100.0 + kl / 1000.0


def validate_weights(lm, con, kl):
    """Stop with an error unless all weights are valid and sum to 1."""
    if lm <= 0 or con < 0 or kl < 0:
        raise ValueError(f"Invalid weights: lm={lm}, con={con}, kl={kl}")
    s = weights_sum(lm, con, kl)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1: lm={lm} + con={con}/100 + kl={kl}/1000 = {s:.6f}"
        )


def ratio_to_weights(con_mult, kl_mult):
    """Scale the SVEN ratio and convert it to (lm, con, kl) that sums to 1."""
    lm_r = SVEN_RATIO["lm"]
    con_r = SVEN_RATIO["con"] * con_mult
    kl_r = SVEN_RATIO["kl"] * kl_mult
    total = lm_r + con_r + kl_r
    con = int(round(con_r / total * 100))      # steps of 0.01
    kl = int(round(kl_r / total * 1000))       # steps of 0.001
    lm = round(1.0 - con / 100.0 - kl / 1000.0, 3)  # remainder goes to lm
    validate_weights(lm, con, kl)
    return (lm, con, kl)


def multiplier_pairs():
    if SEARCH_MODE == "grid":
        return list(itertools.product(CON_MULTIPLIERS, KL_MULTIPLIERS))
    if SEARCH_MODE == "coordinate_con":
        return [(c, 1.0) for c in CON_MULTIPLIERS]
    if SEARCH_MODE == "coordinate_kl":
        return [(BEST_CON_MULTIPLIER, k) for k in KL_MULTIPLIERS]
    raise ValueError(f"Unknown SEARCH_MODE: {SEARCH_MODE}")


def normalize_tuple(w):
    lm, con, kl = w
    return (round(float(lm), 3), int(con), int(kl))


def build_weight_candidates():
    if STAGE == "final":
        if not FINAL_WEIGHTS:
            raise ValueError(
                "FINAL_WEIGHTS is empty. Put the top candidates selected in "
                "Stage 2 as (lm, con, kl) tuples."
            )
        cands = [normalize_tuple(w) for w in FINAL_WEIGHTS]
    else:
        cands = [ratio_to_weights(c, k) for c, k in multiplier_pairs()]
        cands += [normalize_tuple(w) for w in REFERENCE_WEIGHTS]

    unique = []
    for w in cands:
        validate_weights(*w)
        if w not in unique:
            unique.append(w)
    return unique


def print_weight_table(cands):
    print("Loss weight candidates (lm + con/100 + kl/1000 = 1):")
    print("     lm     con      kl    sum  |  ratio (LM=1)")
    for lm, con, kl in cands:
        c, k = con / 100.0, kl / 1000.0
        print(f"  {lm:.3f}   {c:.3f}   {k:.3f}  {weights_sum(lm, con, kl):.3f}"
              f"  |  1 : {c / lm:.2f} : {k / lm:.2f}")
    print()


def stage_settings():
    if STAGE == "screen":
        epochs, seeds = SCREEN_EPOCHS, SCREEN_SEEDS
    elif STAGE == "final":
        epochs, seeds = FINAL_EPOCHS, FINAL_SEEDS
    else:
        raise ValueError(f"Unknown STAGE: {STAGE}")
    if not PASS_SEED:
        if len(seeds) > 1:
            print("[WARN] PASS_SEED is False, so multiple seeds would be identical. "
                  "Using a single run per setting.\n")
        seeds = [None]
    return epochs, seeds


# ---------- Run names ----------
def build_run_name(hp: Dict) -> str:
    tgt_tag = hp["lora_target_modules"].replace(",", "+")
    lr_tag = str(hp["learning_rate"])
    name = (
        f"{BASE}-ep{hp['num_train_epochs']}"
        f"-lr{lr_tag}"
        f"_r{hp['lora_r']}_a{hp['lora_alpha']}_ld{hp['lora_dropout']}"
        f"_t{tgt_tag}"
        f"_wu{hp['warmup_steps']}"
        f"_ga{hp['grad_acc_steps']}"
        f"_lm{hp['lm_loss_ratio']:.3f}_con{hp['contrastive_loss_ratio']}"
        f"_kl{hp['kl_loss_ratio']}"
    )
    # Seed suffix only in Stage 3 (Stage 1 names stay compatible with old ones)
    if hp["seed"] is not None and STAGE == "final":
        name += f"_s{hp['seed']}"
    return name


# ---------- Skip check ----------
def run_already_completed(run_dir: Path) -> bool:
    """
    A run counts as completed if checkpoint-epoch-* or checkpoint-last exists.
    A run that only has train.log (crashed run) is trained again.
    """
    if not run_dir.exists():
        return False
    epoch_re = re.compile(r"^checkpoint-epoch-\d+$")
    has_epoch_ckpt = any(
        p.is_dir() and epoch_re.match(p.name) for p in run_dir.iterdir()
    )
    has_last_ckpt = (run_dir / "checkpoint-last").exists()
    return has_epoch_ckpt or has_last_ckpt


# ---------- Cleanup (keep ONLY the final epoch checkpoint) ----------
def cleanup_epoch_checkpoints(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    epoch_re = re.compile(r"^checkpoint-epoch-(\d+)$")
    epoch_ckpts = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = epoch_re.match(p.name)
        if m:
            epoch_ckpts.append((int(m.group(1)), p))
    if not epoch_ckpts:
        return

    final_epoch, final_path = max(epoch_ckpts, key=lambda x: x[0])
    removed = 0
    for _, p in epoch_ckpts:
        if p == final_path:
            continue
        subprocess.run(["rm", "-rf", str(p)], check=False)
        removed += 1

    if removed:
        print(f"[CLEANUP] Removed {removed} intermediate epoch checkpoints under "
              f"{run_dir} (kept checkpoint-epoch-{final_epoch} as final)\n")


# ---------- Run train.py (stream output to terminal and train.log) ----------
def run_train_and_log(cmd: List[str], log_path: Path) -> int:
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
    return process.returncode


# ---------- Write run_config.json ----------
def write_run_config(run_dir: Path, run_name: str, hp: Dict, returncode: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    lm, con, kl = hp["lm_loss_ratio"], hp["contrastive_loss_ratio"], hp["kl_loss_ratio"]
    cfg = {
        "run_name": run_name,
        "stage": STAGE,
        "search_mode": SEARCH_MODE if STAGE == "screen" else None,
        "model_type": MODEL_TYPE,
        "pretrain": PRETRAIN,
        "hyperparameters": hp,
        "loss_weights": {
            "lm": lm,
            "contrastive": con / 100.0,
            "kl": kl / 1000.0,
            "sum": round(weights_sum(lm, con, kl), 6),
            "raw_args": {"lm_loss_ratio": lm,
                         "contrastive_loss_ratio": con,
                         "kl_loss_ratio": kl},
        },
        "selection_criterion": SELECTION_CRITERION,
        "train_returncode": returncode,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ---------- Build run list ----------
def build_runs(grad_acc_steps_values: List[int]) -> List[Dict]:
    epochs, seeds = stage_settings()
    cands = build_weight_candidates()
    print_weight_table(cands)

    runs: List[Dict] = []
    for lr, tgt, r, ld, wu, ga, (lm, con, kl), seed in itertools.product(
        LEARNING_RATES, LORA_TARGETS, LORA_RS, LORA_DROPOUTS,
        WARMUP_STEPS_LIST, grad_acc_steps_values, cands, seeds
    ):
        runs.append({
            "learning_rate": lr,
            "lora_target_modules": tgt,
            "lora_r": r,
            "lora_alpha": 4 * r,          # alpha = 4 * rank
            "lora_dropout": ld,
            "warmup_steps": wu,
            "grad_acc_steps": ga,
            "num_train_epochs": epochs,
            "seed": seed,
            # loss weights (sum = 1)
            "lm_loss_ratio": lm,
            "contrastive_loss_ratio": con,
            "kl_loss_ratio": kl,
        })
    return runs


# ==============================================================
# Main
# ==============================================================
def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

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
    print("Grid: loss-weight search for LoRA (LoRA hyperparameters fixed)")
    print(f"Stage         : {STAGE}")
    print(f"Search mode   : {SEARCH_MODE if STAGE == 'screen' else '(final candidates)'}")
    print(f"Base model    : {PRETRAIN}")
    print(f"Model type    : {MODEL_TYPE}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    print(f"Assumed per-GPU batch: {per_gpu_batch} (for effective-batch print)")
    print(f"Detected n_gpu (from CVD): {ngpu}")
    print(f"grad_acc_steps candidates: {grad_acc_steps_values}")
    print(f"Total runs    : {len(runs)}")
    print("Selection     : functional correctness drop <= epsilon vs. baseline, "
          "then max security rate")
    print("===================================================\n")

    trained, skipped, failed = 0, 0, []

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

            # objective weights (sum = 1)
            "--contrastive_loss_ratio", str(hp["contrastive_loss_ratio"]),
            "--kl_loss_ratio", str(hp["kl_loss_ratio"]),
            "--lm_loss_ratio", str(hp["lm_loss_ratio"]),

            # schedule / training
            "--warmup_steps", str(hp["warmup_steps"]),
            "--num_train_epochs", str(hp["num_train_epochs"]),

            # accumulation
            "--grad_acc_steps", str(hp["grad_acc_steps"]),
        ]
        if hp["seed"] is not None:
            cmd += ["--seed", str(hp["seed"])]

        print("===================================================")
        print(f"[{i:02d}/{len(runs)}] >> Running experiment: {run_name}")
        print("HP:",
              f"lr={hp['learning_rate']}, r={hp['lora_r']}, alpha={hp['lora_alpha']}, "
              f"dropout={hp['lora_dropout']}, target={hp['lora_target_modules']}, "
              f"warmup={hp['warmup_steps']}, grad_acc_steps={hp['grad_acc_steps']}, "
              f"epochs={hp['num_train_epochs']}, seed={hp['seed']}")
        print(f"Loss weights: lm={hp['lm_loss_ratio']:.3f}, "
              f"con={hp['contrastive_loss_ratio'] / 100:.3f}, "
              f"kl={hp['kl_loss_ratio'] / 1000:.3f} "
              f"(sum={weights_sum(hp['lm_loss_ratio'], hp['contrastive_loss_ratio'], hp['kl_loss_ratio']):.3f})")
        print(f"Effective global batch per update (assuming per_gpu=1): {eff_batch}")
        print("Command:", " ".join(cmd))
        print("Saving into:", run_dir)
        print("===================================================\n")

        if SKIP_EXISTING and run_already_completed(run_dir):
            print(f"[INFO] Completed run found; skipping training: {run_dir}\n")
            skipped += 1
            continue

        if DRY_RUN:
            continue

        returncode = run_train_and_log(cmd, log_path)
        write_run_config(run_dir, run_name, hp, returncode)
        if returncode != 0:
            print(f"[ERROR] train.py failed (exit {returncode}): {run_name}\n")
            failed.append(run_name)
        else:
            trained += 1

        cleanup_epoch_checkpoints(run_dir)

    print("===================================================")
    print(f"Done. Trained: {trained} | Skipped (already existed): {skipped} | "
          f"Failed: {len(failed)} | Total: {len(runs)}"
          + ("  [DRY_RUN]" if DRY_RUN else ""))
    for name in failed:
        print(f"  FAILED: {name}")
    print("===================================================")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# ==============================================================
# grid_train_prompt_new.py
#
# Purpose
#   Search the weights of the multi-objective loss
#   (LM loss, contrastive loss, KL loss) for prompt-tuning on CodeGen.
#   Method-specific hyperparameters (learning rate, number of virtual tokens) are
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
# Two modes
# --------------------------------------------------------------
#   1) SELECTED_RUNS mode: put run names in SELECTED_RUNS. Hyperparameters
#      are parsed from the names and only those runs are trained.
#   2) Grid mode: when SELECTED_RUNS = [], candidates are generated as
#      described above.
#
# Run naming convention
#   <base>-lr<lr>_v<v>_lm<lm>_con<con>_kl<kl>_ep<ep>[_s<seed>]
#   e.g. 350m-lr0.4_v320_lm0.148_con61_kl242_ep3
#        350m-lr0.4_v320_lm0.148_con61_kl242_ep7_s2   (Stage 3, with seed)
#
# Resume / skip
#   If SKIP_EXISTING is True, a run is skipped when its directory already
#   holds checkpoint-epoch-* or checkpoint-last. A run directory without any
#   checkpoint (e.g. a crashed run) is trained again.
#
# Disk management
#   After each run, intermediate epoch checkpoints are deleted. Only the
#   final epoch checkpoint, checkpoint-last and non-checkpoint files
#   (e.g. logs) are kept.
#
# Record
#   run_config.json is written into each run directory (stage, actual
#   weights, seed, selection criterion, ...) so that evaluation / aggregation
#   scripts can read it.
#
# Notes on prompt-tuning hyperparameters (vs. prefix-tuning)
#   - Virtual tokens are prepended to the input embeddings instead of
#     prefix vectors at every layer.
#   - The learning rate is high because continuous prompt embeddings need
#     more aggressive optimization.
#   - model_type is "prompt"; the CLI flag is --n_prompt_token.
# ==============================================================

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # limit visible GPUs to avoid mismatch errors

import itertools
import json
import re
import subprocess
from pathlib import Path


# ==============================================================
# 0) Run settings
# ==============================================================
STAGE = "screen"        # "screen" (Stage 1) or "final" (Stage 3)
DRY_RUN = False         # True: only print candidates and commands, no training
SKIP_EXISTING = True    # True: skip runs that are already trained

# ==============================================================
# 1) Explicit run list (optional). If empty, grid mode is used.
# ==============================================================
SELECTED_RUNS = []

# ==============================================================
# 2) Method-specific hyperparameters
#    Keep ONE value each while searching the loss weights,
#    so that only the effect of the weights is observed.
# ==============================================================
learning_rate  = [0.4]    # prompt-tuning learning rate
# learning_rate = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
n_prompt_token = [320]    # number of virtual tokens (v)
# n_prompt_token = [5, 10, 20, 50]

# ==============================================================
# 3) Loss-weight search settings
# ==============================================================
# Ratio of the original SVEN paper (LM = 1)
SVEN_RATIO = {"lm": 1.0, "con": 4.0, "kl": 1.6}

# "grid" | "coordinate_con" | "coordinate_kl" (see header)
SEARCH_MODE = "grid"
CON_MULTIPLIERS = [0.5, 1.0, 2.0]   # multipliers for the contrastive ratio
KL_MULTIPLIERS  = [0.5, 1.0, 2.0]   # multipliers for the KL ratio
BEST_CON_MULTIPLIER = 1.0           # contrastive multiplier fixed in "coordinate_kl"

# Previous setting for comparison: (lm, con, kl). Must sum to 1.
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
# 4) Epochs and seeds
# ==============================================================
SCREEN_EPOCHS = 3          # Stage 1: short training (enough to rank candidates)
FINAL_EPOCHS  = 7          # Stage 3: normal training
SCREEN_SEEDS  = [1]
FINAL_SEEDS   = [1, 2, 3]
PASS_SEED     = True       # set to False if train.py has no --seed option

# ==============================================================
# 5) Common settings
# ==============================================================
pretrain = "Salesforce/codegen-350M-multi"
model_type = "prompt"
base = "350m"
TRAINED_ROOT = Path("../trained")   # output root of train.py (relative to /scripts)


# ==============================================================
# Weight generation and validation
# ==============================================================
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
    con = int(round(con_r / total * 100))          # steps of 0.01
    kl = int(round(kl_r / total * 1000))           # steps of 0.001
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


# ==============================================================
# Run names
# ==============================================================
def build_run_name(lr, v, lm, con, kl, ep, seed=None):
    name = f"{base}-lr{lr}_v{v}_lm{lm:.3f}_con{con}_kl{kl}_ep{ep}"
    # Seed suffix only in Stage 3 (Stage 1 names stay compatible with old ones)
    if seed is not None and STAGE == "final":
        name += f"_s{seed}"
    return name


# Accepted formats (ep and seed are optional):
#   350m-lr0.05_v50_lm0.180_con41_kl410_ep20
#   350m-lr0.05_v50_lm0.180_con41_kl410_ep7_s2
#   350m-lr0.05_v50_lm0.180_con41_kl410        (ep = default of the stage)
RUN_PATTERN = re.compile(
    r"^(?P<base>[^-]+)-"
    r"lr(?P<lr>[\d.eE+-]+)_"
    r"v(?P<v>\d+)_"
    r"lm(?P<lm>[\d.]+)_"
    r"con(?P<con>\d+)_"
    r"kl(?P<kl>\d+)"
    r"(?:_ep(?P<ep>\d+))?"
    r"(?:_s(?P<seed>\d+))?$"
)


def parse_run_name(run_name: str, default_ep: int):
    m = RUN_PATTERN.match(run_name)
    if not m:
        raise ValueError(
            f"Run name does not match expected pattern:\n"
            f"  {run_name}\n\n"
            f"Expected format:\n"
            f"  {base}-lr<lr>_v<v>_lm<lm>_con<con>_kl<kl>_ep<ep>[_s<seed>]\n"
            f"Example:\n"
            f"  350m-lr0.4_v320_lm0.148_con61_kl242_ep7"
        )
    ep = m.group("ep")
    seed = m.group("seed")
    hp = {
        "lr": float(m.group("lr")),
        "v": int(m.group("v")),
        "lm": float(m.group("lm")),
        "con": int(m.group("con")),
        "kl": int(m.group("kl")),
        "ep": int(ep) if ep is not None else default_ep,
        "seed": int(seed) if seed is not None else None,
    }
    # Normalize the name (always include ep; seed comes last)
    name = run_name[: m.end("kl")] + f"_ep{hp['ep']}"
    if hp["seed"] is not None:
        name += f"_s{hp['seed']}"
    return name, hp


# ==============================================================
# Skip check (has this run already been trained?)
# ==============================================================
def run_already_completed(run_dir: Path) -> bool:
    """
    A run counts as completed if its directory exists and contains
    checkpoint-epoch-* or checkpoint-last. A directory without any
    checkpoint (crashed run) is trained again.
    """
    if not run_dir.exists():
        return False
    epoch_re = re.compile(r"^checkpoint-epoch-\d+$")
    has_epoch_ckpt = any(
        p.is_dir() and epoch_re.match(p.name) for p in run_dir.iterdir()
    )
    has_last_ckpt = (run_dir / "checkpoint-last").exists()
    return has_epoch_ckpt or has_last_ckpt


# ==============================================================
# Cleanup (keep ONLY the final epoch checkpoint)
# ==============================================================
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
    else:
        print(f"[CLEANUP] Only the final epoch checkpoint "
              f"(checkpoint-epoch-{final_epoch}) present under {run_dir}; nothing removed\n")


# ==============================================================
# Write run_config.json
# ==============================================================
def write_run_config(run_dir: Path, run_name: str, hp: dict, returncode: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_name": run_name,
        "stage": STAGE,
        "search_mode": SEARCH_MODE if STAGE == "screen" else None,
        "model_type": model_type,
        "pretrain": pretrain,
        "hyperparameters": hp,
        "loss_weights": {
            "lm": hp["lm"],
            "contrastive": hp["con"] / 100.0,
            "kl": hp["kl"] / 1000.0,
            "sum": round(weights_sum(hp["lm"], hp["con"], hp["kl"]), 6),
            "raw_args": {"lm_loss_ratio": hp["lm"],
                         "contrastive_loss_ratio": hp["con"],
                         "kl_loss_ratio": hp["kl"]},
        },
        "selection_criterion": SELECTION_CRITERION,
        "train_returncode": returncode,
    }
    with open(run_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ==============================================================
# Build run list
# ==============================================================
def build_run_list():
    epochs, seeds = stage_settings()
    run_list = []

    if SELECTED_RUNS:
        for raw_name in SELECTED_RUNS:
            name, hp = parse_run_name(raw_name, default_ep=epochs)
            s = weights_sum(hp["lm"], hp["con"], hp["kl"])
            if abs(s - 1.0) > 1e-6:
                print(f"[WARN] Weights of {name} do not sum to 1 (sum={s:.4f}).")
            run_list.append((name, hp))
        return run_list

    cands = build_weight_candidates()
    print_weight_table(cands)
    for lr, v, (lm, con, kl), seed in itertools.product(
        learning_rate, n_prompt_token, cands, seeds
    ):
        hp = {"lr": lr, "v": v, "lm": lm, "con": con, "kl": kl,
              "ep": epochs, "seed": seed}
        name = build_run_name(lr, v, lm, con, kl, epochs, seed)
        run_list.append((name, hp))
    return run_list


# ==============================================================
# Execute
# ==============================================================
def main():
    run_list = build_run_list()

    print("===================================================")
    print(f"Stage        : {STAGE}")
    print(f"Search mode  : {SEARCH_MODE if STAGE == 'screen' else '(final candidates)'}")
    print(f"Model        : {pretrain} ({model_type})")
    print(f"Total runs   : {len(run_list)}")
    print("Selection    : functional correctness drop <= epsilon vs. baseline, "
          "then max security rate")
    print("===================================================\n")

    skipped, trained, failed = 0, 0, []

    for run_name, hp in run_list:
        run_dir = TRAINED_ROOT / run_name

        if SKIP_EXISTING and run_already_completed(run_dir):
            print("===================================================")
            print(f">> Skipping (already trained): {run_name}")
            print(f"   Found existing run at: {run_dir}")
            print("===================================================\n")
            skipped += 1
            continue

        cmd = [
            "python", "train.py",
            "--output_name", run_name,
            "--model_type", model_type,
            "--pretrain_dir", pretrain,
            "--learning_rate", str(hp["lr"]),
            "--n_prompt_token", str(hp["v"]),
            "--lm_loss_ratio", str(hp["lm"]),
            "--contrastive_loss_ratio", str(hp["con"]),
            "--kl_loss_ratio", str(hp["kl"]),
            "--num_train_epochs", str(hp["ep"]),
        ]
        if hp["seed"] is not None:
            cmd += ["--seed", str(hp["seed"])]

        print("===================================================")
        print(f">> Running experiment: {run_name}")
        print("Parsed hyperparameters:", hp)
        print(f"Loss weights : lm={hp['lm']:.3f}, con={hp['con'] / 100:.3f}, "
              f"kl={hp['kl'] / 1000:.3f} "
              f"(sum={weights_sum(hp['lm'], hp['con'], hp['kl']):.3f})")
        print("Command:", " ".join(cmd))
        print("Saving into:", run_dir)
        print("===================================================\n")

        if DRY_RUN:
            continue

        result = subprocess.run(cmd, check=False)
        write_run_config(run_dir, run_name, hp, result.returncode)
        if result.returncode != 0:
            print(f"[ERROR] train.py failed (exit {result.returncode}): {run_name}\n")
            failed.append(run_name)
        else:
            trained += 1

        cleanup_epoch_checkpoints(run_dir)

    print("===================================================")
    print(f"Done. Trained: {trained} | Skipped (already existed): {skipped} | "
          f"Failed: {len(failed)} | Total: {len(run_list)}"
          + ("  [DRY_RUN]" if DRY_RUN else ""))
    for name in failed:
        print(f"  FAILED: {name}")
    print("===================================================")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# ==============================================================
# new_experiment_process_prompt.py
#
# Purpose
#   End-to-end experiment for choosing the weights of the multi-objective
#   loss (LM loss, contrastive loss, KL loss) for PROMPT-TUNING.
#   One command runs everything: training, security evaluation
#   (sec_eval.py), functional evaluation (HumanEval), screening,
#   selection, final training with several seeds, and the final decision.
#
#   Method-specific hyperparameters (learning rate, number of virtual tokens) are
#   fixed. ONLY the loss weights are searched.
#
# Usage (run from the scripts/ directory, like the other grid scripts)
#   python new_experiment_process_prompt.py                 # all stages
#   python new_experiment_process_prompt.py --stage 1       # only Stage 1
#   python new_experiment_process_prompt.py --stage 2       # uses Stage 1 result file
#   python new_experiment_process_prompt.py --stage 3       # uses Stage 2 result file
#   python new_experiment_process_prompt.py --dry_run       # print the plan only
#
#   Check that the result TXT files are parsed correctly (do this FIRST,
#   using TXT files you already have from earlier evaluations):
#   python new_experiment_process_prompt.py --check_parse_sec  <run>-1.txt
#   python new_experiment_process_prompt.py --check_parse_func human-eval-<...>.txt
#
#   The script can be stopped and started again at any time: finished
#   training runs and finished evaluations (existing TXT files) are skipped.
#
# ==============================================================
# How the loss weights are represented (they always sum to 1)
# ==============================================================
#   Value passed to train.py  ->  actual weight
#       lm  : --lm_loss_ratio          (float) -> used as is
#       con : --contrastive_loss_ratio (int)   -> con / 100
#       kl  : --kl_loss_ratio          (int)   -> kl  / 1000
#   Every candidate satisfies  lm + con/100 + kl/1000 = 1.
#   validate_weights() stops the script if this is violated.
#
# ==============================================================
# Overall workflow
# ==============================================================
#
#   Stage 1: train all candidates briefly and screen them by loss
#   -------------------------------------------------------------
#   Candidates
#     - Defined as RATIOS first, then normalized to sum to 1.
#       (If the weights were varied directly under the sum-to-1 constraint,
#        raising one weight would automatically lower the others, and we
#        could not tell which loss caused an effect.)
#     - Center: ratio of the original SVEN paper, LM : con : KL = 1 : 4 : 1.6
#       (= 0.152 : 0.606 : 0.242 when normalized).
#     - The contrastive and KL ratios are each multiplied by 0.5, 1, 2
#       -> 3 x 3 = 9 candidates.
#     - con is rounded to steps of 0.01 and kl to steps of 0.001 (integer
#       format of train.py); the remainder goes to lm so the sum is exactly 1.
#     - The SVEN candidate (contrastive x1, KL x1) is the REFERENCE run.
#       It is the yardstick for screening (and the functional anchor when
#       BASELINE_FUNCTIONAL is None), and it is also a valid candidate.
#       Reason: its ratio is published in the SVEN paper, so the yardstick
#       does not depend on a setting we chose ourselves.
#     - Because con is stored in steps of 0.01, the SVEN run is trained as
#       0.148 : 0.61 : 0.242 (ratio 1 : 4.12 : 1.64), the closest weights
#       that sum exactly to 1.
#     - Our previous self-chosen setting (lm 0.18, con 0.41, kl 0.41) is NOT
#       used as a reference. Set INCLUDE_PREVIOUS_SETTING = True to add it as
#       an ordinary extra candidate for comparison.
#   Training
#     - SCREEN_EPOCHS (3) epochs, training seed SCREEN_TRAIN_SEED (1).
#       Short training is enough to drop clearly bad candidates.
#   Screening (proxy metrics, no generation needed)
#     - Read the LAST "val epoch" line of train.log.
#     - train.log logs WEIGHTED losses (lm + contrastive + kl = loss), so each
#       value is divided by its weight to get the UNWEIGHTED loss.
#       Weighted values cannot be compared across different weights.
#     - For each candidate, relative degradation vs. the SVEN reference run:
#           deg = (loss_candidate - loss_reference) / loss_reference
#       computed for the LM, contrastive and KL losses.
#         LM loss up          -> generates the training code less well
#         KL loss up          -> drifts away from the base model
#                                (risk for functional correctness)
#         contrastive loss up -> separates secure / vulnerable code less well
#                                (risk for security)
#     - A candidate PASSES if every degradation <= LOSS_TOLERANCE (10 %).
#     - At least MIN_KEEP_AFTER_SCREEN (4) and at most MAX_KEEP_AFTER_SCREEN
#       (6) candidates go on; if needed, candidates are ranked by their worst
#       degradation (smaller is better) to fill / cut the list.
#       The SVEN reference run always goes on as well.
#     - Set SCREEN_BY_LOSS = False to skip screening (all candidates go on).
#
#   Stage 2: light evaluation of the remaining candidates and selection
#   -------------------------------------------------------------------
#   Evaluation (on the Stage 1 models)
#     - Security rate: sec_eval.py + print_results.py, temperature SEC_TEMP.
#     - Functional correctness: human_eval_gen.py + human_eval_exec.py +
#       print_results.py --eval_type human_eval.
#       HE_NUM_SAMPLES (100) completions are generated per problem and the
#       metric pass@FUNC_K (pass@100, as in all earlier experiments) is
#       computed from them. Tables show this as
#       "pass@100 (100 samples/problem)".
#     - "Light" = only SCREEN_EVAL_TRIALS (2) trials per model
#       (trial i uses generation seed i). The settings of each trial are the
#       same as in the full evaluation, so the numbers are comparable.
#   Selection criterion (the same criterion is used again in Stage 3)
#     1) Functional filter: keep candidates with
#            mean pass@100 >= FUNC_ANCHOR - EPSILON
#        FUNC_ANCHOR = BASELINE_FUNCTIONAL (pass@100 of the model WITHOUT
#        fine-tuning, measured earlier) if it is set, otherwise the pass@100
#        of the SVEN reference run. EPSILON = 1.0 point (HumanEval has 164
#        problems, so 1 point is about 1-2 problems).
#     2) Rank the remaining candidates by mean security rate (higher is
#        better; ties broken by higher pass@100).
#     3) The top TOP_K (2) go to Stage 3.
#     - If NO candidate passes the functional filter, the TOP_K candidates
#       with the highest pass@100 are taken instead (a warning is printed).
#     - The weighted validation loss is NOT used for selection: changing
#       the weights changes the scale of that metric itself, so a larger LM
#       weight always looks better (this is why the earlier grid search
#       stuck to the upper bound of the LM weight).
#     - Security rate alone is NOT used either: it would favor the
#       strongest contrastive loss even if the generated code stops working.
#
#   Stage 3: full training with several seeds and the final decision
#   ------------------------------------------------------------------
#     - Each finalist is trained with FINAL_EPOCHS (7) epochs and with
#       training seeds FINAL_TRAIN_SEEDS (1, 2, 3).
#       (A training seed fixes the random parts of training: initial values
#        of the added parameters, data order, dropout. Training with several
#        seeds shows how much the result changes by chance.)
#     - Each model is evaluated with FINAL_EVAL_TRIALS (5) trials.
#     - For each finalist: mean over trials per training seed, then
#       mean and standard deviation over the training seeds.
#     - The same selection criterion as Stage 2 is applied.
#     - If the security-rate difference between the best two eligible
#       finalists is not larger than their standard deviation, they are
#       treated as equal and the one with the higher pass@100 is chosen
#       (decided in advance to avoid choosing after seeing the results).
#     - The SVEN reference run is also trained in Stage 3 (for comparison)
#       if INCLUDE_REFERENCE_IN_FINAL is True, or automatically if
#       BASELINE_FUNCTIONAL is None (it is then needed as the anchor).
#
# ==============================================================
# Outputs (in ../pipeline_results/prompt/)
# ==============================================================
#   stage1_screening.json / .txt  : weights, unweighted val losses, screening
#   stage2_selection.json / .txt  : light-evaluation results, finalists
#   stage3_final.json     / .txt  : full results and the chosen weights
#   sec_eval_txt/                 : TXT outputs of print_results.py (security)
#   human_eval_txt/               : TXT outputs of print_results.py (HumanEval)
#   The heavy folders under ../experiments/ are deleted after each TXT is
#   written, as in the existing grid evaluation scripts.
#   Each trained run directory also gets run_config.json.
#
# ==============================================================
# Things to check before the first full run
# ==============================================================
#   1) Parsing of the print_results.py output: run --check_parse_sec and
#      --check_parse_func on existing TXT files and confirm that the printed
#      numbers match the overall security rate and pass@100. If not, edit
#      SEC_RATE_REGEX / SEC_OVERALL_REGEX / FUNC_REGEX.
#   2) BASELINE_FUNCTIONAL: set to the pass@100 of the model without
#      fine-tuning measured with the same HumanEval settings, or None.
#   3) Run once with --dry_run and check the commands.
# ==============================================================

import argparse
import itertools
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


# >>> METHOD-SPECIFIC BEGIN ===================================
# Settings that differ between prefix / lora / prompt
# ==============================================================
METHOD = "prompt"
MODEL_TYPE = "prompt"
PRETRAIN = "Salesforce/codegen-350M-multi"
BASE_TAG = "350m"
CUDA_VISIBLE_DEVICES = "0"      # None: leave the environment unchanged

# Fixed method hyperparameters (not searched)
LEARNING_RATE = 0.4             # prompt-tuning learning rate
N_PROMPT_TOKEN = 640            # number of virtual tokens

# Directory (inside ../trained/<run_name>/) passed to --model_dir in the
# evaluations. CHECK this against your prompt-tuning evaluation script:
# if prompt models are loaded from another folder, change it here.
EVAL_MODEL_SUBDIR = "checkpoint-last"


def method_train_args():
    """Method-specific arguments for train.py."""
    return [
        "--learning_rate", str(LEARNING_RATE),
        "--n_prompt_token", str(N_PROMPT_TOKEN),
    ]


def method_run_tag():
    """Method-specific part of the run name."""
    return f"lr{LEARNING_RATE}_v{N_PROMPT_TOKEN}"
# <<< METHOD-SPECIFIC END =====================================


# ==============================================================
# Loss-weight candidates
# ==============================================================
SVEN_RATIO = {"lm": 1.0, "con": 4.0, "kl": 1.6}   # LM = 1
CON_MULTIPLIERS = [0.5, 1.0, 2.0]
KL_MULTIPLIERS = [0.5, 1.0, 2.0]
# Reference run = the SVEN candidate (contrastive x1, KL x1); see the header.
REFERENCE_MULTIPLIERS = (1.0, 1.0)                # (contrastive, KL)

# Previous self-chosen setting: NOT a reference. Optional extra candidate.
INCLUDE_PREVIOUS_SETTING = False
PREVIOUS_WEIGHTS = (0.180, 41, 410)               # (lm, con, kl)

# Set to a list of (lm, con, kl) to skip Stage 1/2 selection and use these
# as the Stage 3 finalists, e.g. [(0.148, 61, 242), (0.120, 49, 390)].
FINAL_WEIGHTS_OVERRIDE = None

# ==============================================================
# Training schedule
# ==============================================================
SCREEN_EPOCHS = 3
FINAL_EPOCHS = 7
SCREEN_TRAIN_SEED = 1
FINAL_TRAIN_SEEDS = [1]

# ==============================================================
# Stage 1: screening by unweighted validation loss
# ==============================================================
SCREEN_BY_LOSS = True
LOGGED_LOSSES_ARE_WEIGHTED = True     # train.log: lm + contrastive + kl = loss
LOSS_TOLERANCE = {"lm": 0.10, "contrastive": 0.10, "kl": 0.10}
MIN_KEEP_AFTER_SCREEN = 4             # not counting the SVEN reference run
MAX_KEEP_AFTER_SCREEN = 6

# ==============================================================
# Evaluation settings (same as the existing grid evaluation scripts)
# ==============================================================
SEC_TEMP = 0.4
HE_MAX_GEN_LEN = 200
# Number of completions generated PER PROBLEM (passed as --num_samples).
# This is NOT the metric itself: pass@k is computed FROM these samples.
# pass@100 = share of problems for which at least one of the 100 samples
# passes the tests. The generation time is proportional to this number.
HE_NUM_SAMPLES = 100
HE_NUM_SAMPLES_PER_GEN = 10           # completions per generate() call
# Functional metric: pass@FUNC_K. 100 = pass@100, the same metric as in all
# earlier experiments (e.g. 14.97 % without fine-tuning).
# pass@k needs k <= HE_NUM_SAMPLES. If FUNC_K is changed, BASELINE_FUNCTIONAL
# must be changed to the baseline of the same metric.
FUNC_K = 100
HE_CONTROL = "sec"
SCREEN_EVAL_TRIALS = 2                # Stage 2 (light evaluation)
FINAL_EVAL_TRIALS = 5                 # Stage 3 (full evaluation)

# ==============================================================
# Selection
# ==============================================================
# pass@FUNC_K (%) = pass@100 of the model WITHOUT fine-tuning, measured earlier
# with the same HumanEval settings. None -> the SVEN reference run is used as
# the anchor.
BASELINE_FUNCTIONAL = 14.97
EPSILON = 2.0                          # allowed drop of pass@FUNC_K (points)
TOP_K = 2                              # finalists for Stage 3
INCLUDE_REFERENCE_IN_FINAL = False     # forced to True if BASELINE_FUNCTIONAL is None

# ==============================================================
# Parsing of print_results.py output (check with --check_parse_*)
# ==============================================================
# Security: a line matching SEC_OVERALL_REGEX (or one of the next 3 lines)
# must contain SEC_RATE_REGEX. If no such line exists, the mean of all
# SEC_RATE_REGEX matches is used and a note is printed.
SEC_OVERALL_REGEX = r"overall"
SEC_RATE_REGEX = r"sec[_ ]?rate\D{0,20}?(\d+(?:\.\d+)?)"
# Functional: pass@FUNC_K (the lookahead (?!\d) avoids matching a longer
# number, e.g. pass@10 or pass@100 when FUNC_K = 1)
FUNC_REGEX = rf"pass@{FUNC_K}(?!\d)\D{{0,20}}?(\d+(?:\.\d+)?)"

# Label used in all printed tables, e.g. "pass@100 (100 samples/problem)"
FUNC_LABEL = f"pass@{FUNC_K} ({HE_NUM_SAMPLES} samples/problem)"
FUNC_SHORT = f"pass@{FUNC_K}"
if FUNC_K > HE_NUM_SAMPLES:
    raise ValueError(f"pass@{FUNC_K} needs at least {FUNC_K} samples per problem "
                     f"(HE_NUM_SAMPLES = {HE_NUM_SAMPLES}).")
if HE_NUM_SAMPLES % HE_NUM_SAMPLES_PER_GEN != 0:
    raise ValueError("HE_NUM_SAMPLES must be a multiple of HE_NUM_SAMPLES_PER_GEN "
                     "(required by human_eval_gen.py).")

# ==============================================================
# Paths (relative to scripts/)
# ==============================================================
TRAINED_ROOT = Path("../trained")
EXPERIMENTS_ROOT = Path("../experiments")
RESULTS_ROOT = Path("../pipeline_results") / METHOD
SEC_TXT_DIR = RESULTS_ROOT / "sec_eval_txt"
FUNC_TXT_DIR = RESULTS_ROOT / "human_eval_txt"

DRY_RUN = False   # set by --dry_run


# ==============================================================
# Small helpers
# ==============================================================
def banner(text):
    print("\n" + "=" * 62)
    print(text)
    print("=" * 62 + "\n", flush=True)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_text(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def mean_or_none(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def stdev_or_zero(values):
    values = [v for v in values if v is not None]
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def fmt(x, nd=2):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def run_cmd(cmd, stdout_path=None, tee_path=None):
    """
    Run a command (list form, no shell). Returns True on success.
    stdout_path: write stdout to this file (written via a .tmp file, so a
                 failed command never leaves a file that looks finished).
    tee_path   : print stdout and also save it to this file.
    """
    print("[RUN] " + " ".join(cmd), flush=True)
    if DRY_RUN:
        return True

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = stdout_path.with_name(stdout_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f).returncode
        if rc == 0:
            tmp.replace(stdout_path)
        elif tmp.exists():
            tmp.unlink()
        return rc == 0

    if tee_path is not None:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tee_path, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
            proc.wait()
        return proc.returncode == 0

    return subprocess.run(cmd).returncode == 0


# ==============================================================
# Loss weights
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
            f"Weights must sum to 1: lm={lm} + con={con}/100 + kl={kl}/1000 = {s:.6f}")


def ratio_to_weights(con_mult, kl_mult):
    """Scale the SVEN ratio and convert it to (lm, con, kl) that sums to 1."""
    lm_r = SVEN_RATIO["lm"]
    con_r = SVEN_RATIO["con"] * con_mult
    kl_r = SVEN_RATIO["kl"] * kl_mult
    total = lm_r + con_r + kl_r
    con = int(round(con_r / total * 100))           # steps of 0.01
    kl = int(round(kl_r / total * 1000))            # steps of 0.001
    lm = round(1.0 - con / 100.0 - kl / 1000.0, 3)  # remainder goes to lm
    validate_weights(lm, con, kl)
    return lm, con, kl


def make_candidate(lm, con, kl, label, is_reference=False):
    lm, con, kl = round(float(lm), 3), int(con), int(kl)
    validate_weights(lm, con, kl)
    return {"lm": lm, "con": con, "kl": kl, "label": label,
            "is_reference": is_reference}


def weight_key(c):
    return (c["lm"], c["con"], c["kl"])


def build_stage1_candidates():
    cands, seen = [], set()
    ref = sven_reference_candidate()
    for cm, km in itertools.product(CON_MULTIPLIERS, KL_MULTIPLIERS):
        lm, con, kl = ratio_to_weights(cm, km)
        c = make_candidate(lm, con, kl, f"con x{cm:g}, kl x{km:g}")
        if weight_key(c) == weight_key(ref):
            c = ref                      # the SVEN run is the reference
        if weight_key(c) not in seen:
            cands.append(c)
            seen.add(weight_key(c))
    if weight_key(ref) not in seen:      # if the grid does not contain x1, x1
        cands.append(ref)
        seen.add(weight_key(ref))
    if INCLUDE_PREVIOUS_SETTING:
        prev = make_candidate(*PREVIOUS_WEIGHTS, label="previous (own)")
        if weight_key(prev) not in seen:
            cands.append(prev)
    return cands


def sven_reference_candidate():
    """The SVEN-ratio candidate used as the reference run."""
    lm, con, kl = ratio_to_weights(*REFERENCE_MULTIPLIERS)
    return make_candidate(lm, con, kl, label="SVEN 1:4:1.6 (ref)", is_reference=True)


def weights_str(c):
    return (f"lm={c['lm']:.3f} con={c['con'] / 100:.3f} kl={c['kl'] / 1000:.3f} "
            f"(sum={weights_sum(c['lm'], c['con'], c['kl']):.3f})")


# ==============================================================
# Training
# ==============================================================
def build_run_name(c, epochs, seed=None):
    name = (f"{BASE_TAG}-{method_run_tag()}"
            f"_lm{c['lm']:.3f}_con{c['con']}_kl{c['kl']}_ep{epochs}")
    if seed is not None:
        name += f"_s{seed}"
    return name


def run_already_completed(run_dir):
    if not run_dir.exists():
        return False
    epoch_re = re.compile(r"^checkpoint-epoch-\d+$")
    has_epoch = any(p.is_dir() and epoch_re.match(p.name) for p in run_dir.iterdir())
    return has_epoch or (run_dir / "checkpoint-last").exists()


def cleanup_epoch_checkpoints(run_dir):
    """Delete intermediate epoch checkpoints; keep the final one."""
    if not run_dir.exists():
        return
    epoch_re = re.compile(r"^checkpoint-epoch-(\d+)$")
    ckpts = []
    for p in run_dir.iterdir():
        m = epoch_re.match(p.name)
        if p.is_dir() and m:
            ckpts.append((int(m.group(1)), p))
    if not ckpts:
        return
    final_epoch, final_path = max(ckpts, key=lambda x: x[0])
    for _, p in ckpts:
        if p != final_path:
            shutil.rmtree(p, ignore_errors=True)


def train(c, epochs, seed, add_seed_to_name):
    """Train one run (skipped if already completed). Returns run name or None."""
    name = build_run_name(c, epochs, seed if add_seed_to_name else None)
    run_dir = TRAINED_ROOT / name

    if run_already_completed(run_dir):
        print(f"[SKIP] already trained: {name}")
        return name

    cmd = ["python", "train.py",
           "--output_name", name,
           "--model_type", MODEL_TYPE,
           "--pretrain_dir", PRETRAIN]
    cmd += method_train_args()
    cmd += ["--lm_loss_ratio", str(c["lm"]),
            "--contrastive_loss_ratio", str(c["con"]),
            "--kl_loss_ratio", str(c["kl"]),
            "--num_train_epochs", str(epochs),
            "--seed", str(seed)]

    print(f"[TRAIN] {name}\n        weights: {weights_str(c)}")
    if DRY_RUN:
        run_cmd(cmd)
        return name

    run_dir.mkdir(parents=True, exist_ok=True)
    ok = run_cmd(cmd, tee_path=run_dir / "pipeline_stdout.log")
    save_json(run_dir / "run_config.json", {
        "run_name": name, "method": METHOD, "model_type": MODEL_TYPE,
        "pretrain": PRETRAIN, "epochs": epochs, "train_seed": seed,
        "loss_weights": {"lm": c["lm"], "contrastive": c["con"] / 100.0,
                         "kl": c["kl"] / 1000.0,
                         "sum": round(weights_sum(c["lm"], c["con"], c["kl"]), 6)},
        "raw_args": {"lm_loss_ratio": c["lm"], "contrastive_loss_ratio": c["con"],
                     "kl_loss_ratio": c["kl"]},
        "train_ok": ok,
    })
    cleanup_epoch_checkpoints(run_dir)
    if not ok or not run_already_completed(run_dir):
        print(f"[ERROR] training failed: {name}")
        return None
    return name


# ==============================================================
# Validation losses from train.log
# ==============================================================
VAL_LINE_RE = re.compile(
    r"val epoch (\d+): lm_loss: ([\d.eE+-]+), contrastive_loss: ([\d.eE+-]+), "
    r"kl_loss: ([\d.eE+-]+), loss: ([\d.eE+-]+)")


def read_last_val_losses(run_dir):
    """Return the losses of the last 'val epoch' line, or None."""
    for log_name in ("train.log", "pipeline_stdout.log"):
        path = run_dir / log_name
        if not path.exists():
            continue
        matches = VAL_LINE_RE.findall(path.read_text(encoding="utf-8", errors="replace"))
        if matches:
            ep, lm, con, kl, loss = matches[-1]
            return {"epoch": int(ep), "lm": float(lm), "contrastive": float(con),
                    "kl": float(kl), "loss": float(loss)}
    return None


def unweighted_losses(logged, c):
    total = logged["lm"] + logged["contrastive"] + logged["kl"]
    if abs(total - logged["loss"]) > 2e-3 * max(1.0, logged["loss"]):
        print(f"[WARN] lm + contrastive + kl ({total:.4f}) != loss ({logged['loss']:.4f}). "
              "Check LOGGED_LOSSES_ARE_WEIGHTED.")
    if not LOGGED_LOSSES_ARE_WEIGHTED:
        return {k: logged[k] for k in ("lm", "contrastive", "kl")}
    w = {"lm": c["lm"], "contrastive": c["con"] / 100.0, "kl": c["kl"] / 1000.0}
    return {k: (logged[k] / w[k] if w[k] > 0 else None) for k in w}


# ==============================================================
# Parsing of print_results.py output
# ==============================================================
def parse_sec_rate(text):
    """Return (security rate, how it was found)."""
    rate_re = re.compile(SEC_RATE_REGEX, re.IGNORECASE)
    overall_re = re.compile(SEC_OVERALL_REGEX, re.IGNORECASE)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if overall_re.search(line):
            for nxt in lines[i:i + 4]:
                m = rate_re.search(nxt)
                if m:
                    return float(m.group(1)), "overall line"
    values = [float(m.group(1)) for line in lines for m in rate_re.finditer(line)]
    if values:
        return statistics.mean(values), f"mean of {len(values)} entries (no overall line)"
    return None, "not found"


def parse_func(text):
    m = re.search(FUNC_REGEX, text, re.IGNORECASE)
    return (float(m.group(1)), FUNC_LABEL) if m else (None, "not found")


# ==============================================================
# Evaluation (same commands as the existing grid evaluation scripts)
# ==============================================================
def eval_security(run_name, trial):
    """sec_eval.py -> print_results.py -> TXT. Returns the TXT path or None."""
    out_name = f"{run_name}-{trial}"
    txt = SEC_TXT_DIR / f"{out_name}.txt"
    if txt.exists() and txt.stat().st_size > 0:
        print(f"[SKIP] security TXT exists: {txt.name}")
        return txt

    model_dir = TRAINED_ROOT / run_name / EVAL_MODEL_SUBDIR
    if not DRY_RUN and not model_dir.exists():
        print(f"[ERROR] missing model dir: {model_dir}")
        return None

    ok = run_cmd(["python", "sec_eval.py",
                  "--model_type", MODEL_TYPE,
                  "--model_dir", str(model_dir),
                  "--output_name", out_name,
                  "--pretrain_dir", PRETRAIN,
                  "--temp", str(SEC_TEMP),
                  "--seed", str(trial)])
    eval_folder = EXPERIMENTS_ROOT / "sec_eval" / out_name
    if not ok or (not DRY_RUN and not eval_folder.exists()):
        print(f"[ERROR] sec_eval failed: {out_name}")
        return None

    ok = run_cmd(["python", "print_results.py", "--eval_dir", str(eval_folder)],
                 stdout_path=txt)
    if not DRY_RUN:
        shutil.rmtree(eval_folder, ignore_errors=True)
    return txt if ok else None


def eval_functional(run_name, trial):
    """human_eval_gen.py -> human_eval_exec.py -> print_results.py -> TXT."""
    out_name = f"{run_name}-{HE_CONTROL}-trial{trial}-seed{trial}"
    txt = FUNC_TXT_DIR / f"human-eval-{out_name}.txt"
    if txt.exists() and txt.stat().st_size > 0:
        print(f"[SKIP] HumanEval TXT exists: {txt.name}")
        return txt

    model_dir = TRAINED_ROOT / run_name / EVAL_MODEL_SUBDIR
    if not DRY_RUN and not model_dir.exists():
        print(f"[ERROR] missing model dir: {model_dir}")
        return None

    ok = run_cmd(["python", "human_eval_gen.py",
                  "--model_type", MODEL_TYPE,
                  "--model_dir", str(model_dir),
                  "--control", HE_CONTROL,
                  "--pretrain_dir", PRETRAIN,
                  "--output_name", out_name,
                  "--max_gen_len", str(HE_MAX_GEN_LEN),
                  "--num_samples", str(HE_NUM_SAMPLES),
                  "--num_samples_per_gen", str(HE_NUM_SAMPLES_PER_GEN),
                  "--seed", str(trial)])
    ok = ok and run_cmd(["python", "human_eval_exec.py", "--output_name", out_name])
    eval_folder = EXPERIMENTS_ROOT / "human_eval" / out_name
    if not ok or (not DRY_RUN and not eval_folder.exists()):
        print(f"[ERROR] HumanEval failed: {out_name}")
        return None

    ok = run_cmd(["python", "print_results.py", "--eval_type", "human_eval",
                  "--eval_dir", str(eval_folder)], stdout_path=txt)
    if not DRY_RUN:
        shutil.rmtree(eval_folder, ignore_errors=True)
    return txt if ok else None


def evaluate_run(run_name, n_trials):
    """Evaluate one trained model n_trials times. Returns per-trial values."""
    sec_vals, func_vals = [], []
    for t in range(1, n_trials + 1):
        print(f"\n--- evaluate {run_name}  trial {t}/{n_trials} ---")
        s_txt = eval_security(run_name, t)
        f_txt = eval_functional(run_name, t)
        if DRY_RUN:
            continue
        s = parse_sec_rate(s_txt.read_text(encoding="utf-8"))[0] if s_txt else None
        f = parse_func(f_txt.read_text(encoding="utf-8"))[0] if f_txt else None
        if s_txt and s is None:
            print(f"[ERROR] security rate not found in {s_txt} (check SEC_RATE_REGEX)")
        if f_txt and f is None:
            print(f"[ERROR] {FUNC_SHORT} not found in {f_txt} (check FUNC_REGEX)")
        sec_vals.append(s)
        func_vals.append(f)
    return {"sec_trials": sec_vals, "func_trials": func_vals,
            "sec_mean": mean_or_none(sec_vals), "func_mean": mean_or_none(func_vals)}


# ==============================================================
# Selection criterion (shared by Stage 2 and Stage 3)
# ==============================================================
def select_candidates(entries, anchor, k):
    """
    entries: dicts with "sec_mean" and "func_mean".
    Keep func_mean >= anchor - EPSILON, rank by sec_mean (ties: func_mean).
    Returns (chosen, note).
    """
    valid = [e for e in entries if e.get("sec_mean") is not None
             and e.get("func_mean") is not None]
    threshold = anchor - EPSILON
    for e in valid:
        e["passes_func_filter"] = e["func_mean"] >= threshold
    eligible = [e for e in valid if e["passes_func_filter"]]
    if eligible:
        ranked = sorted(eligible, key=lambda e: (-e["sec_mean"], -e["func_mean"]))
        return ranked[:k], (f"{FUNC_SHORT} >= {threshold:.2f} (anchor {anchor:.2f} - "
                            f"epsilon {EPSILON}); ranked by security rate")
    ranked = sorted(valid, key=lambda e: -e["func_mean"])
    print(f"[WARN] no candidate reaches {FUNC_SHORT} >= {threshold:.2f}; "
          f"taking the candidates with the highest {FUNC_SHORT} instead.")
    return ranked[:k], "FALLBACK: no candidate passed the functional filter"


# ==============================================================
# Stage 1
# ==============================================================
def stage1():
    banner(f"STAGE 1 ({METHOD}): train {SCREEN_EPOCHS} epochs and screen by loss")
    cands = build_stage1_candidates()
    for c in cands:
        print(f"  {c['label']:<22} {weights_str(c)}")

    for c in cands:
        c["run_name"] = train(c, SCREEN_EPOCHS, SCREEN_TRAIN_SEED, add_seed_to_name=False)
        if DRY_RUN or c["run_name"] is None:
            continue
        logged = read_last_val_losses(TRAINED_ROOT / c["run_name"])
        c["val_logged"] = logged
        c["val_unweighted"] = unweighted_losses(logged, c) if logged else None

    if DRY_RUN:
        print("\n[DRY_RUN] Stage 1 stops here (no losses to screen).")
        return None

    ref = next(c for c in cands if c["is_reference"])
    others = [c for c in cands if not c["is_reference"]]
    usable = [c for c in others if c.get("val_unweighted")]
    for c in others:
        if not c.get("val_unweighted"):
            c["screen"] = "dropped: training failed or no val loss in train.log"

    if not SCREEN_BY_LOSS or not ref.get("val_unweighted"):
        if SCREEN_BY_LOSS:
            print("[WARN] SVEN reference run has no val losses; screening skipped.")
        kept = usable
        for c in kept:
            c["screen"] = "kept (no screening)"
    else:
        r = ref["val_unweighted"]
        for c in usable:
            u = c["val_unweighted"]
            deg = {k: (u[k] - r[k]) / r[k] for k in LOSS_TOLERANCE if u[k] is not None and r[k]}
            c["degradation"] = deg
            c["worst_over_tol"] = max(deg[k] - LOSS_TOLERANCE[k] for k in deg)
            c["passes_screen"] = c["worst_over_tol"] <= 0
        ranked = sorted(usable, key=lambda c: c["worst_over_tol"])
        kept = [c for c in ranked if c["passes_screen"]]
        if len(kept) < MIN_KEEP_AFTER_SCREEN:
            kept = ranked[:MIN_KEEP_AFTER_SCREEN]
        kept = kept[:MAX_KEEP_AFTER_SCREEN]
        for c in usable:
            if c in kept:
                c["screen"] = "kept" if c["passes_screen"] else "kept (filled up to MIN_KEEP)"
            else:
                c["screen"] = "dropped"
    ref["screen"] = "kept (reference)"
    kept = kept + [ref]

    lines = [f"STAGE 1 RESULT ({METHOD}) - unweighted validation losses of the last epoch",
             f"reference = SVEN ratio run; tolerance vs. reference: {LOSS_TOLERANCE}",
             "",
             f"{'label':<22} {'lm':>5} {'con':>5} {'kl':>6} | {'LM':>7} {'CON':>7} {'KL':>7}"
             f" | {'dLM':>6} {'dCON':>6} {'dKL':>6} | screen"]
    for c in cands:
        u = c.get("val_unweighted") or {}
        d = c.get("degradation") or {}
        lines.append(
            f"{c['label']:<22} {c['lm']:5.3f} {c['con'] / 100:5.2f} {c['kl'] / 1000:6.3f} | "
            f"{fmt(u.get('lm'), 4):>7} {fmt(u.get('contrastive'), 4):>7} {fmt(u.get('kl'), 4):>7} | "
            f"{fmt(d.get('lm') and d['lm'] * 100, 1):>6} {fmt(d.get('contrastive') and d['contrastive'] * 100, 1):>6} "
            f"{fmt(d.get('kl') and d['kl'] * 100, 1):>6} | {c.get('screen', '')}")
    lines.append("(dLM / dCON / dKL: % change vs. reference; negative = better)")
    save_text(RESULTS_ROOT / "stage1_screening.txt", lines)
    save_json(RESULTS_ROOT / "stage1_screening.json",
              {"candidates": cands, "kept": [c["run_name"] for c in kept]})
    return kept


# ==============================================================
# Stage 2
# ==============================================================
def stage2(kept):
    banner(f"STAGE 2 ({METHOD}): light evaluation ({SCREEN_EVAL_TRIALS} trials) and selection")
    for c in kept:
        c.update(evaluate_run(c["run_name"], SCREEN_EVAL_TRIALS))
    if DRY_RUN:
        print("\n[DRY_RUN] Stage 2 stops here.")
        return None

    ref = next((c for c in kept if c["is_reference"]), None)
    anchor = BASELINE_FUNCTIONAL
    anchor_src = "BASELINE_FUNCTIONAL (no fine-tuning)"
    if anchor is None:
        anchor = ref["func_mean"] if ref and ref.get("func_mean") is not None else None
        anchor_src = "SVEN reference run (Stage 2)"
    if anchor is None:
        raise RuntimeError("No functional anchor: set BASELINE_FUNCTIONAL.")

    finalists, note = select_candidates(kept, anchor, TOP_K)

    lines = [f"STAGE 2 RESULT ({METHOD}) - {SCREEN_EVAL_TRIALS} trials per model",
             f"functional metric: {FUNC_LABEL}",
             f"anchor {FUNC_SHORT} = {anchor:.2f} from {anchor_src}; epsilon = {EPSILON}",
             f"criterion: {note}", "",
             f"{'label':<22} {'lm':>5} {'con':>5} {'kl':>6} | {'sec':>6} {FUNC_SHORT:>8} | filter | finalist"]
    for c in sorted(kept, key=lambda c: -(c.get("sec_mean") or -1)):
        lines.append(
            f"{c['label']:<22} {c['lm']:5.3f} {c['con'] / 100:5.2f} {c['kl'] / 1000:6.3f} | "
            f"{fmt(c.get('sec_mean')):>6} {fmt(c.get('func_mean')):>8} | "
            f"{'pass' if c.get('passes_func_filter') else 'FAIL':>6} | "
            f"{'YES' if c in finalists else ''}")
    save_text(RESULTS_ROOT / "stage2_selection.txt", lines)
    save_json(RESULTS_ROOT / "stage2_selection.json",
              {"anchor": anchor, "anchor_source": anchor_src, "epsilon": EPSILON,
               "note": note, "evaluated": kept,
               "finalists": [{k: f[k] for k in ("lm", "con", "kl", "label", "is_reference")}
                             for f in finalists],
               "reference": ({k: ref[k] for k in ("lm", "con", "kl", "label", "is_reference")}
                             if ref else None)})
    return finalists, ref


# ==============================================================
# Stage 3
# ==============================================================
def stage3(finalists, ref):
    include_ref = INCLUDE_REFERENCE_IN_FINAL or BASELINE_FUNCTIONAL is None
    # Fresh copies: only the weights and labels (no Stage 2 numbers)
    base_keys = ("lm", "con", "kl", "label", "is_reference")
    models = [{k: f[k] for k in base_keys} for f in finalists]
    if include_ref and ref is not None and all(weight_key(m) != weight_key(ref) for m in models):
        r = {k: ref[k] for k in base_keys}
        r["only_for_comparison"] = BASELINE_FUNCTIONAL is not None
        models.append(r)

    banner(f"STAGE 3 ({METHOD}): {FINAL_EPOCHS} epochs x seeds {FINAL_TRAIN_SEEDS}, "
           f"{FINAL_EVAL_TRIALS} evaluation trials each")
    for m in models:
        m["per_seed"] = []
        for seed in FINAL_TRAIN_SEEDS:
            name = train(m, FINAL_EPOCHS, seed, add_seed_to_name=True)
            res = {"train_seed": seed, "run_name": name}
            if name is not None:
                res.update(evaluate_run(name, FINAL_EVAL_TRIALS))
            m["per_seed"].append(res)
        if DRY_RUN:
            continue
        sec_seed = [r.get("sec_mean") for r in m["per_seed"]]
        func_seed = [r.get("func_mean") for r in m["per_seed"]]
        m["sec_mean"], m["sec_std"] = mean_or_none(sec_seed), stdev_or_zero(sec_seed)
        m["func_mean"], m["func_std"] = mean_or_none(func_seed), stdev_or_zero(func_seed)

    if DRY_RUN:
        print("\n[DRY_RUN] Stage 3 stops here.")
        return None

    anchor, anchor_src = BASELINE_FUNCTIONAL, "BASELINE_FUNCTIONAL (no fine-tuning)"
    if anchor is None:
        r = next(m for m in models if m.get("is_reference"))
        anchor, anchor_src = r["func_mean"], "SVEN reference run (Stage 3)"

    selectable = [m for m in models if not m.get("only_for_comparison")]
    ranked, note = select_candidates(selectable, anchor, len(selectable))
    chosen = ranked[0] if ranked else None
    decision = "best security rate among eligible finalists"
    if len(ranked) >= 2 and not note.startswith("FALLBACK"):
        a, b = ranked[0], ranked[1]
        if a["sec_mean"] - b["sec_mean"] <= max(a["sec_std"], b["sec_std"]):
            chosen = max((a, b), key=lambda m: m["func_mean"])
            decision = ("security-rate difference of the top two "
                        f"({a['sec_mean'] - b['sec_mean']:.2f}) <= their std "
                        f"({max(a['sec_std'], b['sec_std']):.2f}): treated as equal, "
                        f"chose the higher {FUNC_SHORT}")

    lines = [f"STAGE 3 RESULT ({METHOD}) - mean +- std over training seeds {FINAL_TRAIN_SEEDS}",
             f"functional metric: {FUNC_LABEL}",
             f"anchor {FUNC_SHORT} = {anchor:.2f} from {anchor_src}; epsilon = {EPSILON}",
             f"criterion: {note}", "",
             f"{'label':<22} {'lm':>5} {'con':>5} {'kl':>6} | {'sec':>14} | {FUNC_SHORT:>14} | filter"]
    for m in models:
        tag = " (comparison only)" if m.get("only_for_comparison") else ""
        lines.append(
            f"{m['label']:<22} {m['lm']:5.3f} {m['con'] / 100:5.2f} {m['kl'] / 1000:6.3f} | "
            f"{fmt(m.get('sec_mean')):>6} +- {fmt(m.get('sec_std')):>5} | "
            f"{fmt(m.get('func_mean')):>6} +- {fmt(m.get('func_std')):>5} | "
            f"{'pass' if m.get('passes_func_filter') else 'FAIL'}{tag}")
    lines += ["", f"decision: {decision}"]
    if chosen:
        lines.append(f"CHOSEN WEIGHTS: {weights_str(chosen)}  [{chosen['label']}]")
        lines.append(f"  train.py args: --lm_loss_ratio {chosen['lm']} "
                     f"--contrastive_loss_ratio {chosen['con']} --kl_loss_ratio {chosen['kl']}")
    save_text(RESULTS_ROOT / "stage3_final.txt", lines)
    save_json(RESULTS_ROOT / "stage3_final.json",
              {"anchor": anchor, "anchor_source": anchor_src, "epsilon": EPSILON,
               "note": note, "decision": decision, "models": models,
               "chosen": ({k: chosen[k] for k in ("lm", "con", "kl", "label")}
                          if chosen else None)})
    return chosen


# ==============================================================
# Loading results of earlier stages (for --stage 2 / --stage 3)
# ==============================================================
def load_stage1_kept():
    data = load_json(RESULTS_ROOT / "stage1_screening.json")
    kept = set(data["kept"])
    return [c for c in data["candidates"] if c.get("run_name") in kept]


def load_stage2_finalists():
    if FINAL_WEIGHTS_OVERRIDE:
        fins = [make_candidate(*w, label="override") for w in FINAL_WEIGHTS_OVERRIDE]
        return fins, sven_reference_candidate()
    data = load_json(RESULTS_ROOT / "stage2_selection.json")
    return data["finalists"], data.get("reference")


# ==============================================================
# Main
# ==============================================================
def check_parse(paths, kind):
    for p in paths:
        text = Path(p).read_text(encoding="utf-8", errors="replace")
        value, how = parse_sec_rate(text) if kind == "sec" else parse_func(text)
        print(f"{p}: {kind} = {value}  ({how})")


def main():
    global DRY_RUN
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "1", "2", "3"], default="all")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--check_parse_sec", nargs="+", metavar="TXT")
    ap.add_argument("--check_parse_func", nargs="+", metavar="TXT")
    args = ap.parse_args()

    if args.check_parse_sec or args.check_parse_func:
        check_parse(args.check_parse_sec or [], "sec")
        check_parse(args.check_parse_func or [], "func")
        return

    DRY_RUN = args.dry_run
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    banner(f"new_experiment_process ({METHOD}) | stage={args.stage} | "
           f"dry_run={DRY_RUN}\nresults: {RESULTS_ROOT}")

    if args.stage == "3" or (args.stage == "all" and FINAL_WEIGHTS_OVERRIDE):
        finalists, ref = load_stage2_finalists()
        stage3(finalists, ref)
        return

    kept = stage1() if args.stage in ("all", "1") else load_stage1_kept()
    if args.stage == "1" or kept is None:
        if DRY_RUN and args.stage == "all":
            print("[DRY_RUN] Stages 2 and 3 use the results of Stage 1 and are not "
                  "shown. Their evaluation commands look like:")
            DRY_RUN = True
            evaluate_run(build_run_name(build_stage1_candidates()[0], SCREEN_EPOCHS), 1)
        return

    result = stage2(kept)
    if args.stage == "2" or result is None:
        return
    finalists, ref = result
    chosen = stage3(finalists, ref)
    if chosen:
        banner(f"DONE ({METHOD}). Chosen weights: {weights_str(chosen)}")


if __name__ == "__main__":
    main()
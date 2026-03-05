# ==============================================================
# bayesian_train_lora.py
#
# Purpose
#   Perform hyperparameter optimization for LoRA fine-tuning using
#   Bayesian optimization (Optuna). Instead of testing all combinations
#   like a grid search, this script intelligently selects promising
#   hyperparameter configurations to minimize the validation loss.
#
# Overview
#   Each Optuna trial performs the following steps:
#
#     1. Sample hyperparameters from predefined search spaces
#          - learning rate
#          - LoRA rank (r)
#          - LoRA dropout
#          - LoRA target modules (e.g. qkv_proj, out_proj)
#
#     2. Sample multi-objective loss weights
#          - contrastive_loss_ratio
#          - kl_loss_ratio
#
#        The LM loss weight is computed automatically so that:
#
#            lm + (contrastive / 100) + (kl / 1000) = 1
#
#        If lm becomes negative (invalid weight combination),
#        the trial is pruned and skipped.
#
#     3. Construct a unique run name encoding all hyperparameters
#
#        Example:
#
#        2b-ep5-lr1e-04_r8_a16_ld0.1_tqkv_proj_ga2_lm0.300_con30_kl400
#
#     4. Execute train.py with those parameters
#
#     5. Stream training logs to:
#
#           ../trained/<run_name>/train.log
#
#     6. After training finishes:
#           - delete checkpoint-epoch-* directories to save disk
#           - keep checkpoint-last and train.log
#
#     7. Parse the final validation loss from train.log
#
#     8. Return this validation loss to Optuna as the optimization score
#
# Optimization objective
#   The study minimizes the final validation loss reported by train.py.
#
# Key differences from grid_train_lora.py
#
#   grid_train_lora.py
#       - evaluates ALL combinations
#       - deterministic grid search
#
#   bayesian_train_lora.py
#       - evaluates only N_TRIALS configurations
#       - uses Optuna's TPE sampler to explore promising regions
#
# Configuration parameters
#
#   PRETRAIN_DIR
#       Base model used for fine-tuning.
#
#   BASE_TAG
#       Prefix used when constructing run names.
#
#   NUM_TRAIN_EPOCHS
#       Number of training epochs per trial.
#
#   GRAD_ACC_STEPS
#       Gradient accumulation steps passed to train.py.
#
#   N_TRIALS
#       Number of Optuna trials to execute.
#
# Search spaces
#
#   LoRA hyperparameters
#       LEARNING_RATES
#       LORA_RS
#       LORA_DROPOUTS
#       LORA_TARGET_MODULES
#
#   Loss weights
#       CONTRASTIVE_RATIOS
#       KL_RATIOS
#
#   LM weight is computed automatically so the sum of weights is 1.
#
# Disk management
#
#   After each run:
#       checkpoint-epoch-* directories are deleted to save storage.
#
#   Remaining files:
#       checkpoint-last
#       train.log
#
# Usage
#
#   Run the optimization:
#
#       python bayesian_train_lora.py
#
#   The script will run N_TRIALS experiments and print the best configuration
#   (lowest validation loss) at the end.
#
# Output
#
#   Training outputs are stored under:
#
#       ../trained/<run_name>/
#
#   Optuna results (best parameters and scores) are printed to the terminal.
#
# ==============================================================

import re
import subprocess
from pathlib import Path
import optuna

# ==============================================================
# Base settings (edit as needed)
# ==============================================================

PRETRAIN_DIR = "Salesforce/codegen-2B-multi"
MODEL_TYPE = "lora"
BASE_TAG = "2b"              # used in run_name
NUM_TRAIN_EPOCHS = 5

# ONLY accumulation steps (this exists in your Namespace)
GRAD_ACC_STEPS = 2

# how many Optuna trials
N_TRIALS = 10

# where train.py writes runs (relative to /scripts)
TRAINED_ROOT = Path("../trained")

# ==============================================================
# Search spaces (EDIT THESE LISTS)
# ==============================================================

# ---- LoRA / training hyperparameters ----
LEARNING_RATES = [1e-4]
LORA_RS = [8]
LORA_DROPOUTS = [0.1]
LORA_TARGET_MODULES = [
    "qkv_proj",
]

# ---- Objective weights (RAW ratios passed to train.py) ----
# trainer scaling:
#   contrastive_loss *= con/100
#   kl_loss         *= kl/1000
#   lm_loss         *= lm (direct)

CONTRASTIVE_RATIOS = list(range(25, 42, 2))   # /100  -> 0.25..0.41
KL_RATIOS = list(range(250, 421, 20))         # /1000 -> 0.25..0.41

# ==============================================================
# Log parsing
# ==============================================================

# Matches: "val epoch 5: ... , loss: 0.4992"
_VAL_TOTAL_RE = re.compile(r"val epoch \d+:[^\n]*,\s*loss:\s*([0-9]*\.?[0-9]+)")

def extract_final_val_loss(log_path: Path) -> float:
    """Parse the LAST TOTAL validation loss from train.log."""
    text = log_path.read_text(errors="ignore")
    vals = _VAL_TOTAL_RE.findall(text)
    if not vals:
        raise ValueError(f"No 'val epoch ... , loss:' found in {log_path}")
    return float(vals[-1])

def cleanup_epoch_checkpoints(run_dir: Path) -> None:
    """Delete checkpoint-epoch-* dirs, keep checkpoint-last + train.log."""
    if not run_dir.exists():
        return
    removed = 0
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-epoch-"):
            subprocess.run(["rm", "-rf", str(p)], check=False)
            removed += 1
    if removed:
        print(f"[CLEANUP] Removed {removed} epoch checkpoints under {run_dir}\n")

def run_train_and_log(cmd: list[str], log_path: Path) -> None:
    """Run train.py, streaming stdout to both terminal and train.log."""
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
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

# ==============================================================
# Objective
# ==============================================================

def objective(trial: optuna.Trial) -> float:
    # -----------------------------
    # (A) sample LoRA/training hparams
    # -----------------------------
    lr = trial.suggest_categorical("learning_rate", LEARNING_RATES)
    lora_r = trial.suggest_categorical("lora_r", LORA_RS)
    lora_dropout = trial.suggest_categorical("lora_dropout", LORA_DROPOUTS)
    target_modules = trial.suggest_categorical("lora_target_modules", LORA_TARGET_MODULES)

    # Your rule: alpha = 2 * rank
    lora_alpha = 2 * int(lora_r)

    # -----------------------------
    # (B) sample objective weights
    # -----------------------------
    con = trial.suggest_categorical("contrastive_loss_ratio", CONTRASTIVE_RATIOS)
    kl = trial.suggest_categorical("kl_loss_ratio", KL_RATIOS)

    con_w = con / 100.0
    kl_w = kl / 1000.0
    lm = 1.0 - con_w - kl_w

    if lm < 0:
        raise optuna.TrialPruned(
            f"Infeasible weights: lm={lm:.4f} (ct={con_w:.4f}, kl={kl_w:.4f})"
        )

    # Tag targets safely for path naming
    tgt_tag = target_modules.replace(",", "+").replace("/", "_")

    # Make lr readable & filesystem-safe
    lr_tag = f"{lr:.0e}" if lr < 0.001 else f"{lr}".replace(".", "p")

    # Include accumulation steps in run_name to avoid collisions
    run_name = (
        f"{BASE_TAG}-ep{NUM_TRAIN_EPOCHS}"
        f"-lr{lr_tag}"
        f"_r{lora_r}_a{lora_alpha}_ld{lora_dropout}"
        f"_t{tgt_tag}"
        f"_ga{GRAD_ACC_STEPS}"
        f"_lm{lm:.3f}_con{con}_kl{kl}"
    )

    run_dir = TRAINED_ROOT / run_name
    log_path = run_dir / "train.log"

    cmd = [
        "python", "train.py",
        "--output_name", run_name,
        "--model_type", MODEL_TYPE,
        "--pretrain_dir", PRETRAIN_DIR,

        "--learning_rate", str(lr),

        "--lora_r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
        "--lora_dropout", str(lora_dropout),
        "--lora_target_modules", target_modules,

        "--contrastive_loss_ratio", str(con),
        "--kl_loss_ratio", str(kl),
        "--lm_loss_ratio", str(lm),

        "--num_train_epochs", str(NUM_TRAIN_EPOCHS),

        # ONLY use existing argument from your Namespace
        "--grad_acc_steps", str(GRAD_ACC_STEPS),
    ]

    print("===================================================")
    print(f"▶ Running experiment: {run_name}")
    print("Command:", " ".join(cmd))
    print("Saving into:", f"../trained/{run_name}")
    print("LoRA:", f"lr={lr} r={lora_r} alpha={lora_alpha} dropout={lora_dropout} targets={target_modules}")
    print("Weights:", f"lm={lm:.3f}, ct={con_w:.3f}, kl={kl_w:.3f} (sum={lm+con_w+kl_w:.3f})")
    print("Accum:", f"grad_acc_steps={GRAD_ACC_STEPS}")
    print("===================================================\n")

    # Run (skip if already exists)
    if log_path.exists():
        print(f"[INFO] train.log exists; skipping training: {log_path}\n")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_train_and_log(cmd, log_path)

    cleanup_epoch_checkpoints(run_dir)

    # Score = final val loss
    val_loss = extract_final_val_loss(log_path)

    # Save helpful info
    trial.set_user_attr("run_name", run_name)
    trial.set_user_attr("lm_weight", lm)
    trial.set_user_attr("ct_weight", con_w)
    trial.set_user_attr("kl_weight", kl_w)
    trial.set_user_attr("final_val_loss", val_loss)

    # Store chosen hparams too
    trial.set_user_attr("learning_rate", lr)
    trial.set_user_attr("lora_r", lora_r)
    trial.set_user_attr("lora_alpha", lora_alpha)
    trial.set_user_attr("lora_dropout", lora_dropout)
    trial.set_user_attr("lora_target_modules", target_modules)

    # Store accumulation too
    trial.set_user_attr("grad_acc_steps", GRAD_ACC_STEPS)

    print(f"[RESULT] trial={trial.number} final_val_loss={val_loss:.6f}\n")
    return val_loss

# ==============================================================
# Main
# ==============================================================

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Optional: enqueue one starting point (if it exists in your spaces)
    # Here: lr=1e-4, r=8 -> alpha=16, dropout=0.1, targets=out_proj, con=30, kl=400
    if (1e-4 in LEARNING_RATES and 8 in LORA_RS and 0.1 in LORA_DROPOUTS
        and "out_proj" in LORA_TARGET_MODULES and 30 in CONTRASTIVE_RATIOS and 400 in KL_RATIOS):
        study.enqueue_trial({
            "learning_rate": 1e-4,
            "lora_r": 8,
            "lora_dropout": 0.1,
            "lora_target_modules": "out_proj",
            "contrastive_loss_ratio": 30,
            "kl_loss_ratio": 400,
        })

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n==================== BEST ====================")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.")
    else:
        bt = study.best_trial
        print("Best value (min final val loss):", study.best_value)
        print("Best params:", study.best_params)
        print("Best run_name:", bt.user_attrs.get("run_name"))
        print("Best weights:",
              "lm=", bt.user_attrs.get("lm_weight"),
              "ct=", bt.user_attrs.get("ct_weight"),
              "kl=", bt.user_attrs.get("kl_weight"))
        print("Best LoRA:",
              "lr=", bt.user_attrs.get("learning_rate"),
              "r=", bt.user_attrs.get("lora_r"),
              "alpha=", bt.user_attrs.get("lora_alpha"),
              "dropout=", bt.user_attrs.get("lora_dropout"),
              "targets=", bt.user_attrs.get("lora_target_modules"))
        print("Accum:",
              "grad_acc_steps=", bt.user_attrs.get("grad_acc_steps"))
    print("=============================================\n")
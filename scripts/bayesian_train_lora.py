import re
import subprocess
from pathlib import Path

import optuna

# ==============================================================
# Bayesian search (keep simple structure like grid_train.py)
# Score = final val loss (parsed from train log)
#
# trainer.py scaling:
#   contrastive_loss *= contrastive_loss_ratio / 100
#   kl_loss         *= kl_loss_ratio / 1000
#   lm_loss         *= lm_loss_ratio
#
# Constraint (effective weights):
#   lm_loss_ratio + (con/100) + (kl/1000) = 1   with lm_loss_ratio >= 0
# ==============================================================

# ==============================================================
# Hyperparameters (RAW ratios passed to train.py)
# Focus search around well-balanced region near (lm,ct,kl) ≈ (0.34,0.33,0.33)
# ==============================================================

learning_rates = [1e-4]
lora_rs = [8]

contrastive_ratios = list(range(25, 42, 2))   # 25,27,...,41  -> 0.25..0.41
kl_ratios = list(range(250, 421, 20))         # 250,270,...,410 -> 0.25..0.41

# ==============================================================
# Base settings
# ==============================================================

pretrain = "Salesforce/codegen-2B-multi"
model_type = "lora"
base = "2b"
num_train_epochs = "5"

# how many runs (attempts)
N_TRIALS = 30

trained_root = Path("../trained")


# ==============================================================
# Helpers
# ==============================================================

def extract_final_val_loss(log_path: Path) -> float:
    """
    Parse the LAST TOTAL validation loss from a line like:
      "val epoch 5: ... , loss: 0.4992"
    """
    text = log_path.read_text(errors="ignore")

    # IMPORTANT: match ", loss:" (the final total loss), not "lm_loss:"
    vals = re.findall(r"val epoch \d+:[^\n]*,\s*loss:\s*([0-9]*\.?[0-9]+)", text)

    if not vals:
        raise ValueError(f"No 'val epoch ... , loss:' found in {log_path}")

    return float(vals[-1])

def cleanup_epoch_checkpoints(run_dir: Path) -> None:
    """
    Delete checkpoint-epoch-* directories to save disk, but keep:
      - checkpoint-last
      - train.log
    """
    if not run_dir.exists():
        return

    removed = 0
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("checkpoint-epoch-"):
            subprocess.run(["rm", "-rf", str(p)], check=False)
            removed += 1

    if removed > 0:
        print(f"[CLEANUP] Removed {removed} epoch checkpoints under {run_dir}\n")

# ==============================================================
# Objective: choose params -> run train.py -> return final val loss
# ==============================================================

def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_categorical("learning_rate", learning_rates)
    r = trial.suggest_categorical("lora_r", lora_rs)
    con = trial.suggest_categorical("contrastive_loss_ratio", contrastive_ratios)
    kl = trial.suggest_categorical("kl_loss_ratio", kl_ratios)

    # enforce: lm + con/100 + kl/1000 = 1
    con_w = con / 100.0
    kl_w = kl / 1000.0
    lm = 1.0 - con_w - kl_w

    if lm < 0:
        raise optuna.TrialPruned(
            f"Infeasible weights: lm={lm:.4f} (con={con_w:.4f}, kl={kl_w:.4f})"
        )

    run_name = f"{base}-lr{lr}_r{r}_lm{lm:.3f}_con{con}_kl{kl}"
    run_dir = trained_root / run_name
    log_path = run_dir / "train.log"

    cmd = [
        "python", "train.py",
        "--output_name", run_name,
        "--model_type", model_type,
        "--pretrain_dir", pretrain,
        "--learning_rate", str(lr),
        "--lora_r", str(r),
        "--contrastive_loss_ratio", str(con),
        "--kl_loss_ratio", str(kl),
        "--lm_loss_ratio", str(lm),
        "--num_train_epochs", num_train_epochs,
    ]

    print("===================================================")
    print(f"▶ Running experiment: {run_name}")
    print("Command:", " ".join(cmd))
    print("Saving into:", f"../trained/{run_name}")
    print("Weights:",
          f"lm={lm:.3f}, ct={con_w:.3f}, kl={kl_w:.3f} (sum={lm+con_w+kl_w:.3f})")
    print("===================================================\n")

    # run training (skip if already done and train.log exists)
    if log_path.exists():
        print(f"[INFO] train.log exists; skipping training: {log_path}\n")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="")   # ターミナルに表示
                f.write(line)         # train.log に保存
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)


    # NEW: delete checkpoint-epoch-* dirs (keep checkpoint-last + train.log)
    cleanup_epoch_checkpoints(run_dir)
    
    # score = final val loss
    val_loss = extract_final_val_loss(log_path)

    trial.set_user_attr("run_name", run_name)
    trial.set_user_attr("lm_weight", lm)
    trial.set_user_attr("ct_weight", con_w)
    trial.set_user_attr("kl_weight", kl_w)
    trial.set_user_attr("final_val_loss", val_loss)

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

    # Start from (lm, ct, kl) ≈ (0.34, 0.33, 0.33)
    # ct=33 -> 0.33, kl=330 -> 0.33, lm=1-0.33-0.33=0.34
    # IMPORTANT: these values must be included in candidate lists above.
    if 33 in contrastive_ratios and 330 in kl_ratios:
        study.enqueue_trial({
            "learning_rate": 1e-4,
            "lora_r": 8,
            "contrastive_loss_ratio": 33,
            "kl_loss_ratio": 330,
        })

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n==================== BEST ====================")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.")
    else:
        print("Best value (min val loss):", study.best_value)
        print("Best params:", study.best_params)
        print("Best run_name:", study.best_trial.user_attrs.get("run_name"))
        print("Best weights:",
              "lm=", study.best_trial.user_attrs.get("lm_weight"),
              "ct=", study.best_trial.user_attrs.get("ct_weight"),
              "kl=", study.best_trial.user_attrs.get("kl_weight"))
    print("=============================================\n")

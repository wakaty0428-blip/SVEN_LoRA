import re
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# Base directory of this script (absolute path)
BASE_DIR = Path(__file__).resolve().parent

# Path to your training run directory
run_dir = BASE_DIR / "../trained/6b-ep10-lr0.0001_r4_a4_ld0.1_tout_proj_wu0_ga2_lm0.330_con34_kl330"
# ==========================================================

log_path = run_dir / "train.log"

if not log_path.exists():
    raise FileNotFoundError(f"train.log not found in: {log_path}")

print(f"Loading log file: {log_path}")

# === Output folder setup ===
output_root = BASE_DIR / "loss_figures"
run_name = run_dir.name
output_dir = output_root / run_name
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Regex patterns
# ==========================================================
train_pattern = re.compile(
    r"epochs:\s*(\d+)/(\d+),\s*steps:\s*(\d+)/(\d+),\s*"
    r"lm_loss:\s*([\d\.]+),\s*contrastive_loss:\s*([\d\.]+),\s*"
    r"kl_loss:\s*([\d\.]+),\s*loss:\s*([\d\.]+)"
)

val_pattern = re.compile(
    r"val epoch\s*(\d+):\s*"
    r"lm_loss:\s*([\d\.]+),\s*contrastive_loss:\s*([\d\.]+),\s*"
    r"kl_loss:\s*([\d\.]+),\s*loss:\s*([\d\.]+)"
)

# ==========================================================
# Data containers
# ==========================================================
train_epochs, train_lm, train_con, train_kl, train_total = [], [], [], [], []
val_epochs, val_lm, val_con, val_kl, val_total = [], [], [], [], []

# ==========================================================
# Parse log
# ==========================================================
with open(log_path, "r") as f:
    for line in f:
        m_tr = train_pattern.search(line)
        if m_tr:
            ep_now = int(m_tr.group(1))     # e.g., 2
            ep_total = int(m_tr.group(2))   # e.g., 5
            step_now = int(m_tr.group(3))   # e.g., 700
            step_total = int(m_tr.group(4)) # e.g., 2590 (Total across all epochs)

            # --- KEY CALCULATION CHANGE ---
            # steps per epoch = 2590 / 5 = 518
            steps_per_epoch = step_total / ep_total
            
            # calculate steps relative to the START of the current epoch
            # e.g., Step 700 in Epoch 2 is (700 - 518) = 182 steps into Ep 2
            step_in_current_epoch = step_now - ((ep_now - 1) * steps_per_epoch)
            
            # fractional epoch = (2 - 1) + (182 / 518) = 1.35
            x_epoch = (ep_now - 1) + (step_in_current_epoch / steps_per_epoch)
            # ------------------------------

            train_epochs.append(x_epoch)
            train_lm.append(float(m_tr.group(5)))
            train_con.append(float(m_tr.group(6)))
            train_kl.append(float(m_tr.group(7)))
            train_total.append(float(m_tr.group(8)))
            continue

        m_val = val_pattern.search(line)
        if m_val:
            val_epochs.append(int(m_val.group(1)))
            val_lm.append(float(m_val.group(2)))
            val_con.append(float(m_val.group(3)))
            val_kl.append(float(m_val.group(4)))
            val_total.append(float(m_val.group(5)))

# ==========================================================
# Plot
# ==========================================================
plt.figure(figsize=(12, 8))

# --- Color palette (paper-friendly) ---
COLOR_LM  = "#1f77b4"   # blue
COLOR_CON = "#ff7f0e"   # orange
COLOR_KL  = "#2ca02c"   # green
COLOR_TOT = "#d62728"   # red

# ==========================================================
# TRAIN curves (solid lines)
# ==========================================================
plt.plot(train_epochs, train_lm,
         label="Train LM",
         color=COLOR_LM,
         linestyle="-",
         linewidth=1.5)

plt.plot(train_epochs, train_con,
         label="Train Contrastive",
         color=COLOR_CON,
         linestyle="-",
         linewidth=1.5)

plt.plot(train_epochs, train_kl,
         label="Train KL",
         color=COLOR_KL,
         linestyle="-",
         linewidth=1.5)

plt.plot(train_epochs, train_total,
         label="Train Total",
         color=COLOR_TOT,
         linestyle="-",
         linewidth=3)

# ==========================================================
# VALIDATION curves (dashed lines + markers)
# ==========================================================
if len(val_epochs) > 0:
    plt.plot(val_epochs, val_lm,
             label="Val LM",
             color=COLOR_LM,
             linestyle="--",
             marker="o",
             markersize=6,
             linewidth=2)

    plt.plot(val_epochs, val_con,
             label="Val Contrastive",
             color=COLOR_CON,
             linestyle="--",
             marker="s",
             markersize=6,
             linewidth=2)

    plt.plot(val_epochs, val_kl,
             label="Val KL",
             color=COLOR_KL,
             linestyle="--",
             marker="^",
             markersize=6,
             linewidth=2)

    plt.plot(val_epochs, val_total,
             label="Val Total",
             color=COLOR_TOT,
             linestyle="--",
             marker="D",
             markersize=6,
             linewidth=2)

# ==========================================================
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
# plt.title(f"Loss Curves: {run_name}", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=10, loc="upper right", frameon=True)
plt.tight_layout()

# Save
save_path = output_dir / "training_validation_losses.png"
plt.savefig(save_path, dpi=300)
plt.close()

print(f"✅ Saved correctly styled graph to: {save_path}")

import re
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==============================
# Parse txt table into DataFrame
# ==============================
def parse_txt(filepath: str) -> pd.DataFrame:
    """
    Parse the table-like txt file and return a pandas DataFrame.

    Extracts:
      - cwe
      - scenario
      - control
      - sec_rate(mean)
    """
    pattern = re.compile(
        r"\|\s*(cwe-[0-9]+|overall)\s*\|\s*([\w\-]+)\s*\|\s*(\w+)\s*\|\s*([\d\.]+),\s*[\d\.]+,\s*[\d\.]+\s*\|"
    )

    data = []
    with open(filepath, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                cwe, scenario, control, sec_rate = m.groups()
                data.append(
                    {
                        "cwe": cwe,
                        "scenario": scenario,
                        "control": control,
                        "sec_rate": float(sec_rate),
                    }
                )

    return pd.DataFrame(data)


# ==============================
# Build merged DataFrame by mode
# ==============================
def build_df_all(
    mode: str,
    lm_txt: str,
    prefix_txt: str | None,
    lora_sec_txt: str | None,
    lora_vul_txt: str | None
) -> pd.DataFrame:
    """
    Returns df_all with columns:
      - cwe
      - scenario
      - sec_rate
      - model  ('lm', 'sec', 'vul')
    """
    # --- LM baseline ---
    df_lm = parse_txt(lm_txt)
    df_lm["model"] = "lm"

    if mode == "prefix":
        if not prefix_txt:
            raise ValueError("mode=prefix requires --prefix_txt")
        df_pref = parse_txt(prefix_txt)
        df_pref["model"] = df_pref["control"]  # sec / vul
        df_all = pd.concat([df_lm, df_pref], ignore_index=True)

    elif mode == "lora":
        if not lora_sec_txt or not lora_vul_txt:
            raise ValueError("mode=lora requires --lora_sec_txt and --lora_vul_txt")

        df_sec = parse_txt(lora_sec_txt)
        df_vul = parse_txt(lora_vul_txt)

        # Force labels for LoRA because files are separated
        df_sec["model"] = "sec"
        df_vul["model"] = "vul"

        df_all = pd.concat([df_lm, df_sec, df_vul], ignore_index=True)

    else:
        raise ValueError("Unknown mode. Use 'prefix' or 'lora'.")

    return df_all[["cwe", "scenario", "sec_rate", "model"]]


# ==============================
# Plot helper: annotate bars
# ==============================
def annotate_bars(bars, fontsize=11):
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 2,
            f"{h:.1f}",           # show 0.0 too
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# ==============================
# Plot per CWE + overall
# ==============================
def plot_per_cwe_and_overall(df_all: pd.DataFrame, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Colors
    COLOR_LM = "#b0b0b0"      # toned-down gray
    COLOR_SEC = "#82d185"     # your green
    COLOR_VUL = "#eb8d8d"     # your red
    EDGE = "#555555"          # outline gray

    # --------------------------
    # 1) Per-CWE plots (exclude overall)
    # --------------------------
    cwes = sorted(c for c in df_all["cwe"].unique() if c != "overall")

    for cwe in cwes:
        subset = df_all[df_all["cwe"] == cwe]
        scenarios = sorted(subset["scenario"].unique())

        x = np.arange(len(scenarios))
        width = 0.25

        def get_rate(sc: str, model: str) -> float:
            v = subset.loc[(subset["scenario"] == sc) & (subset["model"] == model), "sec_rate"]
            return float(v.values[0]) if len(v) else 0.0

        lm_rates = [get_rate(sc, "lm") for sc in scenarios]
        sec_rates = [get_rate(sc, "sec") for sc in scenarios]
        vul_rates = [get_rate(sc, "vul") for sc in scenarios]

        plt.figure(figsize=(10, 6))
        bars_lm = plt.bar(x - width, lm_rates, width, label="LM", color=COLOR_LM, edgecolor=EDGE, linewidth=0.8)
        bars_sec = plt.bar(x,         sec_rates, width, label="SEC", color=COLOR_SEC, edgecolor=EDGE, linewidth=0.8)
        bars_vul = plt.bar(x + width, vul_rates, width, label="VUL", color=COLOR_VUL, edgecolor=EDGE, linewidth=0.8)

        plt.xticks(x, scenarios, fontsize=13)
        plt.yticks(fontsize=13)
        plt.ylim(0, 110)

        plt.xlabel(cwe.upper(), fontsize=15, labelpad=10)
        plt.ylabel("Security Rate (%)", fontsize=15, labelpad=10)
        plt.title(f"{cwe.upper()} — Security Rate Comparison by Scenario", fontsize=16)

        plt.legend(fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        annotate_bars(bars_lm, fontsize=11)
        annotate_bars(bars_sec, fontsize=11)
        annotate_bars(bars_vul, fontsize=11)

        save_path = out_dir / f"{cwe}_{tag}_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {save_path}")

    # --------------------------
    # 2) Overall plot (NEW)
    # --------------------------
    overall = df_all[(df_all["cwe"] == "overall") & (df_all["scenario"] == "overall")]

    def overall_rate(model: str) -> float:
        v = overall.loc[overall["model"] == model, "sec_rate"]
        return float(v.values[0]) if len(v) else 0.0

    lm_o = overall_rate("lm")
    sec_o = overall_rate("sec")
    vul_o = overall_rate("vul")

    plt.figure(figsize=(6, 6))
    x0 = np.array([0])
    width = 0.25

    bars_lm = plt.bar(x0 - width, [lm_o], width, label="LM",  color=COLOR_LM,  edgecolor=EDGE, linewidth=0.8)
    bars_sec = plt.bar(x0,         [sec_o], width, label="SEC", color=COLOR_SEC, edgecolor=EDGE, linewidth=0.8)
    bars_vul = plt.bar(x0 + width, [vul_o], width, label="VUL", color=COLOR_VUL, edgecolor=EDGE, linewidth=0.8)

    plt.xticks(x0, ["OVERALL"], fontsize=13)
    plt.yticks(fontsize=13)
    plt.ylim(0, 110)

    plt.xlabel("OVERALL", fontsize=15, labelpad=10)
    plt.ylabel("Security Rate (%)", fontsize=15, labelpad=10)
    plt.title("OVERALL — Security Rate Comparison", fontsize=16)

    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    annotate_bars(bars_lm, fontsize=12)
    annotate_bars(bars_sec, fontsize=12)
    annotate_bars(bars_vul, fontsize=12)

    save_path = out_dir / f"overall_{tag}_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {save_path}")

    print("🎉 All CWE figures + overall figure generated successfully.")


# ==============================
# CLI
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prefix", "lora"], required=True)
    parser.add_argument("--lm_txt", required=True)
    parser.add_argument("--prefix_txt", default=None)
    parser.add_argument("--lora_sec_txt", default=None)
    parser.add_argument("--lora_vul_txt", default=None)
    parser.add_argument("--out_dir", default="figures_by_cwe")
    parser.add_argument("--tag", default=None)

    args = parser.parse_args()

    tag = args.tag if args.tag is not None else args.mode
    out_dir = Path(args.out_dir)

    df_all = build_df_all(
        mode=args.mode,
        lm_txt=args.lm_txt,
        prefix_txt=args.prefix_txt,
        lora_sec_txt=args.lora_sec_txt,
        lora_vul_txt=args.lora_vul_txt,
    )

    plot_per_cwe_and_overall(df_all, out_dir=out_dir, tag=tag)


if __name__ == "__main__":
    main()

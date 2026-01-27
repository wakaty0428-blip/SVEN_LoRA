#!/usr/bin/env python3
"""
comp_flops.py  (thop + manual missing regions)

Goal:
  Use thop for what it can cover (module-based ops like nn.Linear),
  and add manual analytic terms for regions thop often misses or undercounts:
    - Attention core matmuls: QK^T and Attn@V
    - Prefix effect via K = L + p
    - LoRA delta (extra branch) on selected Linear modules (e.g., qkv_proj)

Conventions:
  - MACs = multiply-accumulate pairs for GEMM-like ops.
  - FLOPs = 2 * MACs (mul + add).
  - We count forward pass only.

Important notes:
  - thop's "Params" output is not always meaningful for Transformers (can be weird).
    This script uses thop primarily for MACs/FLOPs, not params.
  - Depending on the model implementation (fused attention, torch SDPA),
    thop may NOT count attention-core matmuls. That is why we add them manually.
  - We print:
      (A) thop TOTAL MACs/FLOPs
      (B) manual attention-core MACs/FLOPs (QK^T + Attn@V)
      (C) manual LoRA delta MACs/FLOPs (extra branch)
      (D) HYBRID TOTAL = thop_total + manual_attn_core + lora_delta

  If thop already counts attention-core in your environment, HYBRID can double count.
  Use the numbers to sanity-check.

# baseline
    python comp_flops.py --model_type lm \
    --pretrain_dir Salesforce/codegen-350M-multi \
    --batch_size 1 --seq_len 1024 --print_table

# prefix
    python comp_flops.py --model_type prefix \
    --prefix_dir ../trained/<RUN>/checkpoint-last \
    --pretrain_dir Salesforce/codegen-350M-multi \
    --batch_size 1 --seq_len 1024 --print_table

# lora (qkv_proj)
    python comp_flops.py --model_type lora \
    --pretrain_dir Salesforce/codegen-350M-multi \
    --batch_size 1 --seq_len 1024 \
    --lora_rank 8 --lora_targets qkv_proj --print_table

"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from thop import profile


# -----------------------------
# Utilities
# -----------------------------
def human(n: float) -> str:
    units = ["", "K", "M", "G", "T", "P"]
    x = float(n)
    i = 0
    while abs(x) >= 1000.0 and i < len(units) - 1:
        x /= 1000.0
        i += 1
    return f"{x:.3f}{units[i]}"


def macs_to_flops(macs: float) -> float:
    return 2.0 * macs


# -----------------------------
# Read prefix length p
# -----------------------------
def infer_prefix_length(prefix_dir: Path) -> int:
    f = prefix_dir / "prefix_len.json"
    if f.exists():
        obj = json.loads(f.read_text())
        if "p" in obj:
            return int(obj["p"])

    name = prefix_dir.as_posix()
    m = re.search(r"(?:_p|[-]p)(\d+)", name)
    if m:
        return int(m.group(1))

    bin_path = prefix_dir / "pytorch_model.bin"
    if bin_path.exists():
        sd = torch.load(bin_path, map_location="cpu")
        for k, v in sd.items():
            if "prefix_params" in k and hasattr(v, "shape") and len(v.shape) == 3:
                return int(v.shape[1])

    raise RuntimeError(
        f"Could not infer prefix length p from {prefix_dir}. "
        "Add prefix_len.json with {'p': <int>} or ensure run name contains _pXX_."
    )


# -----------------------------
# Model dims
# -----------------------------
@dataclass
class ModelDims:
    n_layers: int
    d_model: int
    n_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int


def get_dims_from_config(cfg) -> ModelDims:
    n_layers = getattr(cfg, "n_layer", None) or getattr(cfg, "num_hidden_layers", None)
    d_model  = getattr(cfg, "n_embd", None)  or getattr(cfg, "hidden_size", None)
    n_heads  = getattr(cfg, "n_head", None)  or getattr(cfg, "num_attention_heads", None)
    vocab    = getattr(cfg, "vocab_size", None)

    if n_layers is None or d_model is None or n_heads is None or vocab is None:
        raise RuntimeError(f"Could not extract dims from config: {cfg.__class__.__name__}")

    head_dim = d_model // n_heads
    d_ff = getattr(cfg, "n_inner", None) or getattr(cfg, "intermediate_size", None)
    if d_ff is None:
        d_ff = 4 * d_model

    return ModelDims(
        n_layers=int(n_layers),
        d_model=int(d_model),
        n_heads=int(n_heads),
        head_dim=int(head_dim),
        d_ff=int(d_ff),
        vocab_size=int(vocab),
    )


# -----------------------------
# Manual pieces (missing regions)
# -----------------------------
def manual_attention_core_macs(dims: ModelDims, *, B: int, L: int, K: int) -> float:
    """
    Attention-core matmuls only:
      QK^T : B*h*L*K*dh = B*L*K*d
      AV   : same
    total = 2 * B * L * K * d
    """
    d = dims.d_model
    return 2.0 * B * L * K * d


def manual_lora_delta_macs(
    dims: ModelDims,
    *,
    B: int,
    L: int,
    lora_rank: int,
    lora_targets: List[str],
) -> float:
    """
    Extra LoRA branch MACs (baseline Wx is counted by thop via nn.Linear):
      For Linear(in_dim -> out_dim), LoRA adds per token:
        r*(in_dim + out_dim) MACs
      Multiply by B*L and number of layers where it appears.
    """
    n = dims.n_layers
    d = dims.d_model
    dff = dims.d_ff
    r = lora_rank

    total = 0.0
    for t in lora_targets:
        t = t.strip()
        if t == "qkv_proj":
            in_dim, out_dim = d, 3 * d
            total += n * (B * L * (r * (in_dim + out_dim)))
        elif t in ("out_proj", "o_proj", "attn_out"):
            in_dim, out_dim = d, d
            total += n * (B * L * (r * (in_dim + out_dim)))
        elif t in ("mlp_in", "fc_in", "up_proj"):
            in_dim, out_dim = d, dff
            total += n * (B * L * (r * (in_dim + out_dim)))
        elif t in ("mlp_out", "fc_out", "down_proj"):
            in_dim, out_dim = dff, d
            total += n * (B * L * (r * (in_dim + out_dim)))
        else:
            raise ValueError(
                f"Unknown lora target '{t}'. Supported: qkv_proj, out_proj(o_proj), mlp_in, mlp_out"
            )
    return total


# -----------------------------
# thop profiling
# -----------------------------
def thop_total_macs_flops(model: torch.nn.Module, input_ids: torch.Tensor) -> tuple[float, float]:
    """
    Returns (MACs, FLOPs) from thop on a real forward.

    NOTE:
      For Transformers, thop may undercount or miss attention-core ops.
      Also, "params" returned by thop is not used here.
    """
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_ids,), verbose=False)
    macs = float(macs)
    flops = macs_to_flops(macs)
    return macs, flops


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["lm", "prefix", "lora"], required=True)

    p.add_argument("--pretrain_dir", type=str, default=None)
    p.add_argument("--prefix_dir", type=str, default=None)

    p.add_argument("--lora_rank", type=int, default=0)
    p.add_argument("--lora_targets", type=str, default="qkv_proj",
                   help="Comma-separated targets: qkv_proj,out_proj,mlp_in,mlp_out")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    p.add_argument("--print_table", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.model_type == "prefix" and not args.prefix_dir:
        raise ValueError("--prefix_dir is required for model_type=prefix")
    if args.model_type in ("lm", "lora") and not args.pretrain_dir:
        raise ValueError("--pretrain_dir is required for model_type=lm or lora")
    if args.model_type == "prefix" and not args.pretrain_dir:
        # We still need base config; easiest is to require pretrain_dir.
        raise ValueError("Prefix: please provide --pretrain_dir for the base model config.")

    cfg = AutoConfig.from_pretrained(args.pretrain_dir)
    dims = get_dims_from_config(cfg)

    B = int(args.batch_size)
    L = int(args.seq_len)

    # Prefix length p and KV length K
    pfx = 0
    if args.model_type == "prefix":
        pfx = infer_prefix_length(Path(args.prefix_dir))
    K = L + pfx

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.pretrain_dir,
        torch_dtype=torch.float16 if args.device == "cuda" else None,
    ).to(device)

    input_ids = torch.zeros((B, L), dtype=torch.long, device=device)

    # (A) thop totals
    th_macs, th_flops = thop_total_macs_flops(model, input_ids)

    # (B) manual attention core (prefix reflected via K)
    attn_core_macs = manual_attention_core_macs(dims, B=B, L=L, K=K)
    attn_core_flops = macs_to_flops(attn_core_macs)

    # (C) LoRA delta (analytic)
    lora_delta_macs = 0.0
    lora_rank = 0
    lora_targets: List[str] = []
    if args.model_type == "lora":
        lora_rank = int(args.lora_rank)
        if lora_rank <= 0:
            raise ValueError("LoRA: provide --lora_rank > 0 (e.g., 8).")
        lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        lora_delta_macs = manual_lora_delta_macs(
            dims, B=B, L=L, lora_rank=lora_rank, lora_targets=lora_targets
        )
    lora_delta_flops = macs_to_flops(lora_delta_macs)

    # Hybrid
    hybrid_macs = th_macs + attn_core_macs + lora_delta_macs
    hybrid_flops = macs_to_flops(hybrid_macs)

    print("=== comp_flops_hybrid_thop.py results (thop + manual missing regions) ===")
    print("Convention: FLOPs = 2 * MACs (1 MAC = mul+add)")
    print("--- context ---")
    print(f"model_type   : {args.model_type}")
    print(f"pretrain_dir : {args.pretrain_dir}")
    print(f"prefix_dir   : {args.prefix_dir}")
    print(f"batch_size   : {B}")
    print(f"seq_len      : {L}")
    print(f"device       : {args.device}")
    print("--- model dims ---")
    print(f"n_layers     : {dims.n_layers}")
    print(f"d_model      : {dims.d_model}")
    print(f"n_heads      : {dims.n_heads}")
    print(f"head_dim     : {dims.head_dim}")
    print(f"d_ff         : {dims.d_ff}")
    print(f"vocab_size   : {dims.vocab_size}")

    if args.model_type == "prefix":
        print("--- prefix ---")
        print(f"prefix_len p : {pfx}  (kv_len K = L+p = {K})")

    if args.model_type == "lora":
        print("--- lora (analytic delta) ---")
        print(f"lora_rank r  : {lora_rank}")
        print(f"lora_targets : {', '.join(lora_targets)}")

    print("--- tool vs manual ---")
    print(f"thop TOTAL          : {human(th_macs)} MACs | {human(th_flops)} FLOPs")
    print(f"manual attn-core    : {human(attn_core_macs)} MACs | {human(attn_core_flops)} FLOPs"
          f"   (QK^T + Attn@V with K={K})")
    if args.model_type == "lora":
        print(f"manual LoRA delta   : {human(lora_delta_macs)} MACs | {human(lora_delta_flops)} FLOPs"
              f"   (extra branch only; baseline counted in thop)")

    print("--- totals ---")
    print(f"HYBRID TOTAL        : {human(hybrid_macs)} MACs | {human(hybrid_flops)} FLOPs")

    if args.print_table:
        print()
        print("Notes:")
        print("- HYBRID may double-count if thop already includes attention-core matmuls in your setup.")
        print("- To check, compare thop TOTAL growth when increasing seq_len;")
        print("  if it grows ~quadratically, thop likely counts attention-core; if mostly linear, it likely misses it.")


if __name__ == "__main__":
    main()

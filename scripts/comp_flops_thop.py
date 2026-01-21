#!/usr/bin/env python3
"""
comp_flops_thop.py

Compare forward compute (MACs/FLOPs) across SVEN model types using thop (pytorch-OpCounter).

Supports:
- lm     : base LM forward
- prefix : forward with prefix past_key_values (prefix ACTIVE)
- lora   : base LM + PEFT adapter loaded from checkpoint-last/{sec|vul}

Outputs:
- MACs, Params (thop)
- optionally FLOPs = 2 * MACs (explicit convention)

Install:
  uv add thop peft

Examples:
  # 1) LM (Salesforce/codegen-2B-multi)
  python comp_flops_thop.py \
    --model_type lm \
    --pretrain_dir Salesforce/codegen-2B-multi \
    --batch_size 1 --seq_len 1024

  # 2) Prefix (checkpoint dir that has lm.txt + pytorch_model.bin)
  python comp_flops_thop.py \
    --model_type prefix \
    --prefix_dir ../trained/<YOUR_PREFIX_RUN>/checkpoint-last \
    --batch_size 1 --seq_len 1024 \
    --control_id 0

  # 3) LoRA (checkpoint-last containing sec/ and vul/)
  python comp_flops_thop.py \
    --model_type lora \
    --pretrain_dir Salesforce/codegen-2B-multi \
    --lora_ckpt_dir ../trained/<YOUR_LORA_RUN>/checkpoint-last \
    --adapter sec \
    --batch_size 1 --seq_len 1024

Notes / caveats:
- thop provides an estimate and can undercount fused/custom attention ops.
- Still useful for consistent relative comparisons under the same input shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
from thop import profile, clever_format

from peft import LoraConfig, get_peft_model

from sven.model import (
    load_model,
    model_from_pretrained,
    config_from_pretrained,
)


# -----------------------------
# Dummy inputs
# -----------------------------
def build_dummy_inputs(tokenizer, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    vocab_size = getattr(tokenizer, "vocab_size", None) or 50257
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------
# thop wrapper (positional args)
# -----------------------------
class ForwardWrapper(torch.nn.Module):
    """
    Wrapper to make a HF/PEFT model compatible with thop:
      forward(input_ids, attention_mask) -> logits

    We also optionally inject prefix past_key_values for prefix tuning.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        is_prefix: bool = False,
        control_id: int = 0,
    ):
        super().__init__()
        self.model = model
        self.is_prefix = is_prefix
        self.control_id = control_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.is_prefix:
            # IMPORTANT: make prefix ACTIVE by providing past_key_values
            control_ids = [self.control_id] * input_ids.shape[0]
            past = self.model.get_past_from_prefix(control_ids)
            out = self.model(
                input_ids=input_ids,
                past_key_values=past,
                use_cache=False,
            )
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        return out.logits


# -----------------------------
# Load LoRA eval model from your custom saved format
# -----------------------------
def load_lora_from_checkpoint(
    pretrain_dir: str,
    lora_ckpt_dir: Path,
    adapter: str,
    device: torch.device,
):
    """
    Your save format (from model.py save_model):
      {lora_ckpt_dir}/sec/adapter_config.json
      {lora_ckpt_dir}/sec/adapter_model.bin
      {lora_ckpt_dir}/vul/adapter_config.json
      {lora_ckpt_dir}/vul/adapter_model.bin
    where "sec" corresponds to adapter_name "default" during training.
    """
    from transformers import AutoTokenizer

    # tokenizer from base LM
    tokenizer = AutoTokenizer.from_pretrained(pretrain_dir)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # base LM
    cfg = config_from_pretrained(pretrain_dir, pretrain_dir)
    base_model = model_from_pretrained(pretrain_dir, "lm", cfg)

    # adapter files
    adapter_dir = lora_ckpt_dir / adapter
    cfg_path = adapter_dir / "adapter_config.json"
    w_path = adapter_dir / "adapter_model.bin"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing: {cfg_path}")
    if not w_path.exists():
        raise FileNotFoundError(f"Missing: {w_path}")

    adapter_cfg_dict = json.loads(cfg_path.read_text())
    lora_cfg = LoraConfig(**adapter_cfg_dict)

    model = get_peft_model(base_model, lora_cfg)

    # map sec->default, vul->vul (to match your trainer usage)
    # Your saved weights include keys like "lora_A.default"/"lora_A.vul", so adapter naming matters.
    if adapter == "vul" and "vul" not in getattr(model, "peft_config", {}):
        model.add_adapter("vul", lora_cfg)

    # load weights saved in adapter_model.bin
    state = torch.load(w_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    # select adapter
    if adapter == "sec":
        model.set_adapter("default")
    else:
        model.set_adapter("vul")

    # move to device
    model.to(device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))

    # (optional) small debug if needed:
    # print(f"[LoRA load] missing={len(missing)}, unexpected={len(unexpected)}")

    return tokenizer, model


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_type", choices=["lm", "prefix", "lora"], required=True)

    # LM + LoRA use base LM
    p.add_argument("--pretrain_dir", type=str, default=None)

    # Prefix uses a checkpoint dir with lm.txt + pytorch_model.bin
    p.add_argument("--prefix_dir", type=str, default=None)

    # LoRA uses checkpoint-last dir containing sec/ and vul/
    p.add_argument("--lora_ckpt_dir", type=str, default=None)
    p.add_argument("--adapter", choices=["sec", "vul"], default="sec")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_gpu", type=int, default=1)

    # Prefix control id
    p.add_argument("--control_id", type=int, default=0)

    # Reporting convention
    p.add_argument("--as_flops", action="store_true", help="Report FLOPs = 2 * MACs convention")

    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    # ---- Load model/tokenizer ----
    is_prefix = False

    if args.model_type == "lm":
        if not args.pretrain_dir:
            raise ValueError("--pretrain_dir is required for model_type=lm")

        loader_args = SimpleNamespace(
            device=device,
            n_gpu=args.n_gpu,
            # unused fields but safe:
            pretrain_dir=args.pretrain_dir,
            n_prefix_token=None,
            dropout=None,
        )
        tokenizer, model, _ = load_model("lm", args.pretrain_dir, False, loader_args)
        model.to(device).eval()

    elif args.model_type == "prefix":
        if not args.prefix_dir:
            raise ValueError("--prefix_dir is required for model_type=prefix")

        loader_args = SimpleNamespace(
            device=device,
            n_gpu=args.n_gpu,
            pretrain_dir=None,
            n_prefix_token=None,
            dropout=None,
        )
        tokenizer, model, _ = load_model("prefix", args.prefix_dir, False, loader_args)
        model.to(device).eval()
        is_prefix = True

    else:  # lora
        if not args.pretrain_dir or not args.lora_ckpt_dir:
            raise ValueError("--pretrain_dir and --lora_ckpt_dir are required for model_type=lora")

        tokenizer, model = load_lora_from_checkpoint(
            pretrain_dir=args.pretrain_dir,
            lora_ckpt_dir=Path(args.lora_ckpt_dir),
            adapter=args.adapter,
            device=device,
        )
        is_prefix = False

    # ---- Dummy inputs ----
    dummy = build_dummy_inputs(tokenizer, args.batch_size, args.seq_len, device)

    # ---- thop profile ----
    wrapped = ForwardWrapper(model, is_prefix=is_prefix, control_id=args.control_id).eval()
    macs, params = profile(wrapped, inputs=(dummy["input_ids"], dummy["attention_mask"]))

    if args.as_flops:
        flops = 2 * macs
        flops_s, params_s = clever_format([flops, params], "%.3f")
        print("=== thop forward estimate ===")
        print("Metric : FLOPs (FLOPs = 2 * MACs convention)")
        print(f"FLOPs  : {flops_s}")
        print(f"Params : {params_s}")
    else:
        macs_s, params_s = clever_format([macs, params], "%.3f")
        print("=== thop forward estimate ===")
        print("Metric : MACs")
        print(f"MACs   : {macs_s}")
        print(f"Params : {params_s}")

    print("--- context ---")
    print(f"model_type  : {args.model_type}")
    print(f"pretrain_dir: {args.pretrain_dir}")
    print(f"prefix_dir  : {args.prefix_dir}")
    print(f"lora_ckpt   : {args.lora_ckpt_dir}")
    print(f"adapter     : {args.adapter}")
    print(f"control_id  : {args.control_id}")
    print(f"batch_size  : {args.batch_size}")
    print(f"seq_len     : {args.seq_len}")
    print(f"device      : {args.device}")


if __name__ == "__main__":
    main()

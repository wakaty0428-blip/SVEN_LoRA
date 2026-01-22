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

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from thop import clever_format

from peft import LoraConfig, get_peft_model

from sven.model import (
    load_model,
    model_from_pretrained,
    config_from_pretrained,
)

# thop internals (version-robust)
try:
    from thop.profile import register_hooks as THOP_REGISTER_HOOKS
except Exception:
    from thop.vision.basic_hooks import register_hooks as THOP_REGISTER_HOOKS


# -----------------------------
# Dummy inputs
# -----------------------------
def build_dummy_inputs(tokenizer, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    vocab_size = getattr(tokenizer, "vocab_size", None) or 50257
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, device=device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# -----------------------------
# thop cleanup (optional but nice)
# -----------------------------
def clear_thop_buffers(model: torch.nn.Module) -> None:
    """
    Remove thop-added buffers if they exist.
    This alone may not fix PEFT shared-module crashes, but it's good hygiene.
    """
    for m in model.modules():
        if hasattr(m, "_buffers"):
            m._buffers.pop("total_ops", None)
            m._buffers.pop("total_params", None)
        if hasattr(m, "total_ops"):
            try:
                delattr(m, "total_ops")
            except Exception:
                pass
        if hasattr(m, "total_params"):
            try:
                delattr(m, "total_params")
            except Exception:
                pass


# -----------------------------
# SAFE thop profile (fixes "total_ops already exists")
# -----------------------------
def profile_safe(model: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
    """
    Safer replacement for thop.profile() that avoids:
        KeyError: "attribute 'total_ops' already exists"
    which can occur when the same module instance is reachable multiple times
    (shared references common in wrappers / PEFT models).

    Returns: (macs, params)
    """
    handler_collection = {}

    custom_ops = {}  # you can add custom op counters later if needed
    register_hooks = THOP_REGISTER_HOOKS

    def add_hooks(m: torch.nn.Module):
        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            fn = custom_ops[m_type]
        elif m_type in register_hooks:
            fn = register_hooks[m_type]

        # Ensure buffers exist WITHOUT crashing if revisited
        if hasattr(m, "_buffers") and "total_ops" in m._buffers:
            m._buffers["total_ops"] = torch.zeros(1, dtype=torch.float64, device=m._buffers["total_ops"].device)
        else:
            try:
                m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
            except KeyError:
                if hasattr(m, "_buffers"):
                    m._buffers["total_ops"] = torch.zeros(1, dtype=torch.float64)
                else:
                    setattr(m, "total_ops", torch.zeros(1, dtype=torch.float64))

        if hasattr(m, "_buffers") and "total_params" in m._buffers:
            m._buffers["total_params"] = torch.zeros(
                1, dtype=torch.float64, device=m._buffers["total_params"].device
            )
        else:
            try:
                m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))
            except KeyError:
                if hasattr(m, "_buffers"):
                    m._buffers["total_params"] = torch.zeros(1, dtype=torch.float64)
                else:
                    setattr(m, "total_params", torch.zeros(1, dtype=torch.float64))

        # Count parameters (thop-style; may overcount for shared weights)
        try:
            m.total_params += torch.DoubleTensor([sum(p.numel() for p in m.parameters())])
        except Exception:
            pass

        # Register counting hook if available
        if fn is not None:
            handler_collection[m] = m.register_forward_hook(fn)

    # Attach hooks
    model.apply(add_hooks)

    # Run one forward
    with torch.no_grad():
        _ = model(*inputs)

    # Sum totals
    total_ops = 0.0
    total_params = 0.0
    for m in model.modules():
        if hasattr(m, "total_ops"):
            try:
                total_ops += float(m.total_ops.item())
            except Exception:
                pass
        if hasattr(m, "total_params"):
            try:
                total_params += float(m.total_params.item())
            except Exception:
                pass

    # Remove hooks
    for _, h in handler_collection.items():
        try:
            h.remove()
        except Exception:
            pass

    return total_ops, total_params


# -----------------------------
# thop wrapper (positional args)
# -----------------------------
class ForwardWrapper(torch.nn.Module):
    """
    Wrapper to make a HF/PEFT model compatible with thop:
      forward(input_ids, attention_mask) -> logits

    We optionally inject prefix past_key_values for prefix tuning.
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
    Your save format (from sven/model.py save_model):
      {lora_ckpt_dir}/sec/adapter_config.json
      {lora_ckpt_dir}/sec/adapter_model.bin
      {lora_ckpt_dir}/vul/adapter_config.json
      {lora_ckpt_dir}/vul/adapter_model.bin

    where "sec" corresponds to adapter_name "default" during training.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrain_dir)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.bos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    cfg = config_from_pretrained(pretrain_dir, pretrain_dir)
    base_model = model_from_pretrained(pretrain_dir, "lm", cfg)

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

    if adapter == "vul" and "vul" not in getattr(model, "peft_config", {}):
        model.add_adapter("vul", lora_cfg)

    state = torch.load(w_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    if adapter == "sec":
        model.set_adapter("default")
    else:
        model.set_adapter("vul")

    model.to(device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_type", choices=["lm", "prefix", "lora"], required=True)

    p.add_argument("--pretrain_dir", type=str, default=None)
    p.add_argument("--prefix_dir", type=str, default=None)

    p.add_argument("--lora_ckpt_dir", type=str, default=None)
    p.add_argument("--adapter", choices=["sec", "vul"], default="sec")

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_gpu", type=int, default=1)

    p.add_argument("--control_id", type=int, default=0)

    # Keep for backward compatibility (now we always print both)
    p.add_argument("--as_flops", action="store_true", help="(ignored) kept for backward compatibility")

    return p.parse_args()


def main():
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    is_prefix = False

    if args.model_type == "lm":
        if not args.pretrain_dir:
            raise ValueError("--pretrain_dir is required for model_type=lm")

        loader_args = SimpleNamespace(
            device=device,
            n_gpu=args.n_gpu,
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

    dummy = build_dummy_inputs(tokenizer, args.batch_size, args.seq_len, device)

    wrapped = ForwardWrapper(model, is_prefix=is_prefix, control_id=args.control_id).eval()

    clear_thop_buffers(wrapped)

    macs, params = profile_safe(wrapped, inputs=(dummy["input_ids"], dummy["attention_mask"]))

    flops = 2.0 * macs  # convention: 1 MAC = 2 FLOPs (mul+add)

    macs_s, _ = clever_format([macs, macs], "%.3f")   # second value ignored; clever_format wants list
    flops_s, _ = clever_format([flops, flops], "%.3f")
    params_s = clever_format([params], "%.3f")[0]

    print("=== thop forward estimate ===")
    print("Convention: FLOPs = 2 * MACs (1 MAC = mul+add)")
    print(f"MACs  : {macs_s}")
    print(f"FLOPs : {flops_s}")

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
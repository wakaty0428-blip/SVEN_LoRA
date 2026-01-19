import os
import sys
import torch
import numpy
import random
import shutil
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from peft import PeftModel, LoraConfig, get_peft_model

from sven.utils import set_seed
from sven.model import load_model, XGLMForCausalLM, GPT2LMHeadCustomModel
from sven.constant import PROMPTS, MODEL_DIRS
from sven.human_eval.problem_yaml import Problem

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--model_type', type=str, choices=['lm', 'prefix', 'text', 'lora'], required=True)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--control', type=str, choices=['sec', 'vul'], default='sec')

    parser.add_argument('--temp', type=float, default=0.4)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=300)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--num_samples_per_gen', type=int, default=25)

    parser.add_argument('--eval_type', type=str, default='human_eval')
    parser.add_argument('--output_dir', type=str, default='../experiments')
    parser.add_argument('--data_dir', type=str, default='../data_eval')
    parser.add_argument('--pretrain_dir', type=str, default='Salesforce/codegen-350M-multi')


    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    assert args.num_samples % args.num_samples_per_gen == 0
    args.output_dir = os.path.join(args.output_dir, args.eval_type)
    args.data_dir = os.path.join(args.data_dir, args.eval_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, args.output_name)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    shutil.copytree(args.data_dir, args.output_dir)

    return args

args = get_args()

def trim_code(completion, stop_tokens):
    for stop_token in stop_tokens:
        if stop_token in completion:
            completion = completion[:completion.find(stop_token)]
    return completion

def main():
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print("Directory does not exist: {}".format(output_dir))
        sys.exit(1)

    problems = list(
        filter(
            lambda f: not f.name.endswith(".results.yaml"),
            sorted(output_dir.glob("*.yaml")),
        )
    )

    if args.model_type in ('lm', 'text'):
        model_dir = '2b' if args.model_dir is None else args.model_dir
        if model_dir in MODEL_DIRS:
            model_dir = MODEL_DIRS[model_dir]
    else:
        assert args.model_dir is not None
        model_dir = args.model_dir

    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # LoRA version (MANUAL + VERIFIED + DEBUG)
    # ============================================================
    if args.model_type == "lora":
        print("\n=== [HumanEval] Loading BASE LM with LoRA support ===")

        # 1. Load base model WITH LoRA patching
        tokenizer, base_model, device = load_model(
            "lora",                   # <<< MUST be "lora"
            args.pretrain_dir,
            False,
            args,
        )

        # 2. Load LoRA config
        adapter_path = model_dir.rstrip("/")
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"adapter_config.json not found: {cfg_path}")

        with open(cfg_path) as f:
            cfg_dict = json.load(f)

        lora_cfg = LoraConfig(**cfg_dict)

        # 3. Create PEFT model (single adapter named "default")
        model = get_peft_model(base_model, lora_cfg)

        # 4. Load adapter weights
        state_path = os.path.join(adapter_path, "adapter_model.bin")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"adapter_model.bin not found: {state_path}")

        print(f"=== [HumanEval] Loading LoRA weights from: {state_path}")
        state = torch.load(state_path, map_location="cpu")

        # 5. Detect adapter tags and normalize to ".default."
        tags = ["sec", "vul", "default"]
        detected_tags = set()
        for k in state.keys():
            for t in tags:
                if f".{t}." in k:
                    detected_tags.add(t)

        print("=== [HumanEval] Detected adapter tags:", detected_tags or "NONE")

        remapped_state = {}
        for k, v in state.items():
            new_k = k
            for t in detected_tags:
                new_k = new_k.replace(f".{t}.", ".default.")
            remapped_state[new_k] = v

        missing, unexpected = model.load_state_dict(remapped_state, strict=False)

        # ====================================================
        # SANITY CHECKS (CRITICAL)
        # ====================================================
        print("\n=== [Sanity Check] Verifying LoRA is ACTIVE ===")

        lora_layer_count = 0
        zero_layers = []

        # NOTE: PEFT wraps model as model.base_model.model
        blocks = model.base_model.model.transformer.h

        for i, block in enumerate(blocks):
            qkv = block.attn.qkv_proj
            if hasattr(qkv, "lora_A"):
                A = qkv.lora_A["default"].weight
                B = qkv.lora_B["default"].weight
                sA = A.abs().sum().item()
                sB = B.abs().sum().item()
                print(f"  Layer {i:02d}: |A|={sA:.6f}, |B|={sB:.6f}")
                lora_layer_count += 1
                if sA == 0.0 or sB == 0.0:
                    zero_layers.append(i)

        if lora_layer_count == 0:
            raise RuntimeError("[FATAL] No LoRA layers found in model.")

        if zero_layers:
            raise RuntimeError(
                f"[FATAL] LoRA layers exist but ZERO weights at layers: {zero_layers}"
            )

        print(f"=== [Sanity Check] LoRA active in {lora_layer_count} layers ✓ ===")

        model.eval()
        model.to(device)

    
    else:
        tokenizer, model, device = load_model('prefix' if args.model_type == 'prefix' else 'lm', model_dir, False, args)
    
        model.eval()

    
    for problem_yaml_path in tqdm(problems):
        with problem_yaml_path.open() as f:
            problem = Problem.load(f)

        prompt = problem.prompt

        if args.model_type == 'text':
            if args.control == 'sec':
                prompt = PROMPTS[0] + prompt
            else:
                prompt = PROMPTS[1] + prompt

        if isinstance(model, GPT2LMHeadCustomModel):
            prompt = prompt.strip()

        inputs = tokenizer(prompt, return_tensors='pt').to(device)

        if isinstance(model, XGLMForCausalLM):
            del inputs['token_type_ids']

        kwargs = dict()
        if args.model_type == 'prefix':
            if args.control == 'sec':
                kwargs['control_id'] = 0
            else:
                kwargs['control_id'] = 1
                
        for i in range(args.num_samples // args.num_samples_per_gen):
            set_seed(args)
            with torch.no_grad():
                samples = model.generate(
                    **inputs,
                    do_sample=True,
                    num_return_sequences=args.num_samples_per_gen,
                    temperature=args.temp,
                    max_new_tokens=args.max_gen_len,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    **kwargs
                )

            for sample in samples.tolist():
                completion = sample[inputs['input_ids'].shape[1]:]
                if tokenizer.eos_token_id in completion:
                    completion = completion[:completion.index(tokenizer.eos_token_id)]
                completion = tokenizer.decode(completion)
                completion = trim_code(completion, problem.stop_tokens)
                problem.completions.append(completion)

            args.seed += 1

        with problem_yaml_path.open("w") as f:
            f.write(Problem.dump(problem))

if __name__ == '__main__':
    main()
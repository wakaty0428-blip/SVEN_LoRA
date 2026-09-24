"""
train_single.py  ->  place at  scripts/train_single.py

Entry point for the single- vs multi-objective ablation.

The PEFT method is selected on the command line with --model_type, and the
loss formulation with --loss_mode:

    --loss_mode single   ->  LM loss only  (Eq. 1)            [new]
    --loss_mode multi    ->  LM + contrastive + KL (SVEN)     [original behavior]

Everything else (data, prefix/prompt length, LR, epochs, seed, ...) is shared,
so running the two modes with otherwise identical flags gives a controlled
comparison of the multi-objective loss against the single-objective loss.

Examples
--------
# Prefix-tuning, single objective
python train_single.py --model_type prefix --loss_mode single \
    --pretrain_dir 2b --output_name prefix_single --seed 1

# Prefix-tuning, multi objective (SVEN-style) for the baseline of the ablation
python train_single.py --model_type prefix --loss_mode multi \
    --pretrain_dir 2b --output_name prefix_multi --seed 1

# LoRA / prompt-tuning, single objective
python train_single.py --model_type lora   --loss_mode single --output_name lora_single
python train_single.py --model_type prompt --loss_mode single --output_name prompt_single

The resulting checkpoints have the same format as those from train.py, so the
existing evaluation scripts (grid_sec_eval_*, grid_human_eval_*, sec_eval.py,
human_eval_gen.py / human_eval_exec.py) can score security rate and functional
correctness without any change.
"""

import os
import argparse

# Original multi-objective trainers.
from sven.trainer import PrefixTrainer, LoraTrainer, PromptTuningTrainer
# New single-objective trainers.
from sven.single_trainer import build_single_trainer
from sven.utils import set_seed, set_logging, set_devices
from sven.constant import MODEL_DIRS

# PEFT methods for which the single-objective ablation is defined.
PEFT_CHOICES = ['prefix', 'lora', 'prompt']

MULTI_TRAINERS = {
    'prefix': PrefixTrainer,
    'lora':   LoraTrainer,
    'prompt': PromptTuningTrainer,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', type=str, required=True)

    parser.add_argument('--data_dir', type=str, default='../data_train_val')
    parser.add_argument('--output_dir', type=str, default='../trained/')

    # --- PEFT method selected on the command line ---
    parser.add_argument('--model_type', type=str, choices=PEFT_CHOICES, default='prefix')

    # --- loss formulation: the axis of this ablation ---
    parser.add_argument('--loss_mode', type=str, choices=['single', 'multi'], default='single',
                        help="'single' = LM loss only (Eq. 1), trained on secure code only; "
                             "'multi' = LM + contrastive + KL (SVEN-style), trained on secure+vul pairs.")
    # single-objective only: whether the LM loss covers the whole secure snippet
    # ('full', the literal reading of Eq. 1) or just the changed tokens ('diff',
    # SVEN's token masking). Ignored in multi mode.
    parser.add_argument('--secure_loss_region', type=str, choices=['full', 'diff'], default='full',
                        help="single mode: 'full' = loss on the whole secure snippet; "
                             "'diff' = loss only on tokens changed vs the vulnerable version.")

    # ===== LoRA hyperparameters =====
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', type=str, default='qkv_proj')  # comma-separated

    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--vul_type', type=str, default=None)

    parser.add_argument('--n_prefix_token', type=int, default=None)
    parser.add_argument('--n_prompt_token', type=int, default=None)

    parser.add_argument('--num_train_epochs', type=int, default=None)
    parser.add_argument('--kl_loss_ratio', type=int, default=None)          # divided by 1000 in trainer
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--contrastive_loss_ratio', type=int, default=400)  # divided by 100 in trainer

    parser.add_argument('--max_num_tokens', type=int, default=1024)
    parser.add_argument('--grad_acc_steps', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--diff_level', type=str, choices=['prog', 'line', 'char', 'mix'], default='mix')
    parser.add_argument('--lm_loss_ratio', type=float, default=1.0)

    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    # ---- resolve pretrain_dir ----
    if args.pretrain_dir is None:
        args.pretrain_dir = '2b'
    if args.pretrain_dir in MODEL_DIRS:
        args.pretrain_dir = MODEL_DIRS[args.pretrain_dir]

    # ---- prefix length defaults ----
    if args.n_prefix_token is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.n_prefix_token = 5
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.n_prefix_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.n_prefix_token = 12
        else:
            assert False, f'no default n_prefix_token for {args.pretrain_dir}'

    # ---- prompt length defaults ----
    if args.n_prompt_token is None and args.model_type == 'prompt':
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.n_prompt_token = 5
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.n_prompt_token = 8
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.n_prompt_token = 12
        else:
            assert False, f'no default n_prompt_token for {args.pretrain_dir}'

    # ---- epoch defaults ----
    if args.num_train_epochs is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.num_train_epochs = 8
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.num_train_epochs = 5
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.num_train_epochs = 5
        else:
            assert False, f'no default num_train_epochs for {args.pretrain_dir}'

    # ---- kl ratio defaults ----
    if args.kl_loss_ratio is None:
        if args.pretrain_dir == 'Salesforce/codegen-350M-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-2B-multi':
            args.kl_loss_ratio = 1600
        elif args.pretrain_dir == 'Salesforce/codegen-6B-multi':
            args.kl_loss_ratio = 2000
        else:
            assert False, f'no default kl_loss_ratio for {args.pretrain_dir}'

    # ---- learning rate defaults ----
    if args.model_type == 'prefix':
        if args.learning_rate is None:
            args.learning_rate = 1e-2
    elif args.model_type == 'prompt':
        if args.learning_rate is None:
            args.learning_rate = 1e-2
    elif args.model_type == 'lora':
        if args.learning_rate is None:
            args.learning_rate = 1e-4

    # In single-objective mode the security terms are unused. Zero them so the
    # log and any downstream code reflect that only the LM term is active.
    if args.loss_mode == 'single':
        args.contrastive_loss_ratio = 0
        args.kl_loss_ratio = 0

    # Keep runs from the two modes side by side without clobbering each other.
    args.output_dir = os.path.join(args.output_dir, args.output_name)
    return args


def build_trainer(args):
    if args.loss_mode == 'single':
        return build_single_trainer(args.model_type, args)
    # multi
    if args.model_type not in MULTI_TRAINERS:
        raise NotImplementedError(f"unknown model_type={args.model_type}")
    return MULTI_TRAINERS[args.model_type](args)


def main():
    args = get_args()
    set_logging(args, os.path.join(args.output_dir, 'train.log'))
    set_devices(args)
    set_seed(args)

    args.logger.info('loss_mode = %s | model_type = %s', args.loss_mode, args.model_type)
    trainer = build_trainer(args)
    trainer.run()


if __name__ == '__main__':
    main()
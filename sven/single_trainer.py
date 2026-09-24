"""
single_trainer.py  ->  place at  sven/single_trainer.py

Pure single-objective (LM-only), secure-only PEFT trainers.

These are drop-in variants of the multi-objective SVEN-style trainers in
`sven/trainer.py`. They differ from the multi-objective path in TWO ways, both
required for a faithful single-objective baseline:

  1. Loss.   Only the language-modeling term is used:

                L_single = -  sum_{(x, y) in D_sec}  log P_theta(y | x; phi)   (Eq. 1)

     The contrastive term and the KL term are removed entirely.

  2. Data.    Training uses SECURE snippets only (control_id == 0, i.e.
     `func_src_after`). The vulnerable side of each diff pair is dropped, so the
     PEFT parameters are fine-tuned purely on secure code. This matches the
     paper's description of the non-SVEN methods: "standard language modeling
     objectives on our curated secure code dataset."

So the controlled comparison is:

        multi  = lm_loss + contrastive_loss + kl_loss ,  trained on secure+vul pairs   (SVEN)
        single = lm_loss ,                                trained on secure code only   (this module)

with everything else identical (PEFT method, prefix/prompt length, LR, epochs,
seed, eval temperature).

By default the LM loss is applied to the WHOLE secure snippet (weights all 1),
which is the literal reading of Eq. (1). Set `--secure_loss_region diff` to
instead keep SVEN's diff-region token masking (loss only on the changed
tokens) if you want that as an additional variant.

The produced checkpoints have the same format as those from train.py, so the
existing evaluation scripts (grid_sec_eval_*, grid_human_eval_*, sec_eval.py,
human_eval_gen.py / human_eval_exec.py) score security rate and functional
correctness unchanged. Evaluation uses the secure side (control_id == 0)
exactly as before; the vulnerable prefix/adapter is simply never trained.

Usage is via scripts/train_single.py, e.g.

    python train_single.py --model_type prefix --loss_mode single ...
    python train_single.py --model_type lora   --loss_mode single ...
    python train_single.py --model_type prompt --loss_mode single ...
"""

from collections import OrderedDict

# Reuse the multi-objective trainers and the shared loss/logit helpers.
from sven.trainer import (
    PrefixTrainer,
    LoraTrainer,
    PromptTuningTrainer,
    get_logits_from_lm,
    get_logits_from_prompt_lm,
    get_logits_from_lora_lm,
    token_weighted_loss,
)
from sven.dataset import PrefixDataset
from sven.constant import BINARY_LABELS, SEC_LABEL


# ---------------------------------------------------------------------------
# Secure-only dataset
# ---------------------------------------------------------------------------
class SecureOnlyDataset(PrefixDataset):
    """Keeps only the secure side (control_id == 0, func_src_after) of each
    diff pair. Vulnerable samples are dropped entirely.

    Loss region is controlled by args.secure_loss_region:
      - 'full' (default): train on the whole secure snippet  -> weights all 1
                          (the literal single-objective LM loss of Eq. 1).
      - 'diff':           keep SVEN's diff-region token masking (loss only on
                          the tokens that changed relative to the vulnerable
                          version).
    """

    def add_data(self, label, src, changes, vul_id, lang):
        # Drop the vulnerable side of every pair.
        if label != SEC_LABEL:
            return

        # Full-snippet LM loss unless diff masking is explicitly requested.
        region = getattr(self.args, 'secure_loss_region', 'full')
        if region == 'full':
            changes = None  # get_tensor -> weights = [1] * len(tokens)

        control_id = BINARY_LABELS.index(label)  # == 0 for secure
        data = self.get_tensor(src, vul_id, control_id, changes)
        if data is not None:
            self.dataset.append(data)


# ---------------------------------------------------------------------------
# Mixin: swap in the secure-only dataset for train and val
# ---------------------------------------------------------------------------
class _SecureOnlyDataMixin:
    def load_dataset(self):
        self.dataset = SecureOnlyDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = SecureOnlyDataset(self.args, self.tokenizer, 'val')


# ---------------------------------------------------------------------------
# Prefix-tuning, single objective, secure only
# ---------------------------------------------------------------------------
class SinglePrefixTrainer(_SecureOnlyDataMixin, PrefixTrainer):

    def step(self, batch):
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        inputs = inputs.to(self.input_device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.input_device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.input_device)  # all 0 (secure) here

        correct_logits, _ = get_logits_from_lm(self.model, inputs, control_ids)
        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio

        return_dict['lm_loss'] = lm_loss.item()
        return_dict['loss'] = lm_loss.item()
        return lm_loss, return_dict


# ---------------------------------------------------------------------------
# Prompt-tuning, single objective, secure only
# ---------------------------------------------------------------------------
class SinglePromptTuningTrainer(_SecureOnlyDataMixin, PromptTuningTrainer):

    def step(self, batch):
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        inputs = inputs.to(self.input_device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.input_device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.input_device)  # all 0 (secure) here

        correct_logits, _ = get_logits_from_prompt_lm(self.model, inputs, control_ids)
        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio

        return_dict['lm_loss'] = lm_loss.item()
        return_dict['loss'] = lm_loss.item()
        return lm_loss, return_dict


# ---------------------------------------------------------------------------
# LoRA, single objective, secure only
# ---------------------------------------------------------------------------
class SingleLoraTrainer(_SecureOnlyDataMixin, LoraTrainer):
    """load_model (inherited) still builds both the 'default' (secure) and
    'vul' adapters. In secure-only single-objective mode every sample has
    control_id == 0, so only the 'default' adapter is ever activated and
    trained; the 'vul' adapter stays at init and is unused. Evaluation uses the
    'default' (secure) adapter exactly as before.
    """

    def step(self, batch):
        return_dict = OrderedDict()
        inputs, weights, control_ids, _ = batch
        inputs = inputs.to(self.input_device)
        shift_inputs = inputs[..., 1:].squeeze(0)
        weights = weights.to(self.input_device)
        shift_weights = weights[..., 1:].squeeze(0)
        control_ids = control_ids.to(self.input_device)
        control_id_int = int(control_ids.item())  # 0 (secure) here

        self._set_adapter_for_control(control_id_int, correct=True)  # 'default'
        correct_logits, _ = get_logits_from_lora_lm(self.model, inputs)
        lm_loss = token_weighted_loss('cross_entropy', correct_logits, shift_inputs, shift_weights)
        lm_loss *= self.args.lm_loss_ratio

        return_dict['lm_loss'] = lm_loss.item()
        return_dict['loss'] = lm_loss.item()
        return lm_loss, return_dict


# ---------------------------------------------------------------------------
# Factory: pick the single-objective trainer by PEFT type string
# ---------------------------------------------------------------------------
SINGLE_TRAINERS = {
    'prefix': SinglePrefixTrainer,
    'prompt': SinglePromptTuningTrainer,
    'lora':   SingleLoraTrainer,
}


def build_single_trainer(model_type, args):
    """Return an instantiated single-objective (secure-only) trainer."""
    if model_type not in SINGLE_TRAINERS:
        raise NotImplementedError(
            f"single-objective training is not defined for model_type='{model_type}'. "
            f"Choose one of {sorted(SINGLE_TRAINERS)}."
        )
    return SINGLE_TRAINERS[model_type](args)
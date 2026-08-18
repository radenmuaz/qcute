"""qcute_v5_stack_noreg/ks1: Ks=(1,), context_len=256, attn_window=-1, code_hard=True/code_sample=False,
entropy_reg_weight=0.0 (disabled) -- simplest rung of the Ks grid, sibling to ks21/ks221 in this
folder, all three sharing entropy_reg_weight=0 as the no-regularization baseline counterpart to
configs/qcute_v5_stack_ks221_ste_entropyreg.py (entropy_reg_weight=0.1). full_val_eval=True: every
--eval_every round scores the WHOLE val set (eval_model_full, batch_size=-1) instead of a sampled
subset.

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/qcute_v5_stack_noreg/ks1.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_noreg_ks1
"""
from pathlib import Path

decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 256
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20
full_val_eval = True

qual_gen_bytes = 128
qual_prompt_bytes = 64

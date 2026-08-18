"""qcute_v5_stack_noreg/ks221: Ks=(2,2,1), context_len=256, attn_window=-1, code_hard=True/code_sample=False,
entropy_reg_weight=0.0 (disabled) -- see ks1.py's docstring for this folder's shared rationale
(no-regularization baseline grid, sibling to ks1/ks21). Same Ks/hyperparams as
configs/qcute_v5_stack_ks221_ste_entropyreg.py except entropy_reg_weight=0.0 here vs 0.1 there --
direct A/B pair (that config also predates full_val_eval, so its own val_bpb numbers were
sampled, not exhaustive; not directly comparable to this run's full-val numbers without a rerun).

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/qcute_v5_stack_noreg/ks221.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_noreg_ks221

# measure code usage after training (add this checkpoint name to CHECKPOINTS in the script first):
uv run python scripts/measure_code_entropy.py qcute_v5_stack_noreg_ks221
"""
from pathlib import Path

decoder_type = "stack"
Ks = (2, 2, 1)
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

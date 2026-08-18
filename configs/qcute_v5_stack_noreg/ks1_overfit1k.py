"""qcute_v5_stack_noreg/ks1_overfit1k: same architecture as ks1.py (Ks=(1,), stack decoder,
context_len=256, code_hard=True/code_sample=False, quant_type="simplex", entropy_reg_weight=0.0) but on a
tiny n_bytes=1000 slice with val_frac=0.5 (500/500 train/val split) -- fast-iteration overfit check
rather than a real training run, same spirit as the project's standard *_overfit10k_*.py testbeds
(see CLAUDE.md's "Standing methodology").

uv run python -m qcute.qcute_v5 --decoder_type stack --config configs/qcute_v5_stack_noreg/ks1_overfit1k.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_noreg_ks1_overfit1k
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
n_bytes = 1000
val_frac = 0.5

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 200
eval_batches = 20
full_val_eval = True

qual_gen_bytes = 128
qual_prompt_bytes = 64

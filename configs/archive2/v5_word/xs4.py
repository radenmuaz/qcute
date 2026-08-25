"""v5_word/xs4: same as xs.py but n_layers=4 -- xs preset's own native depth, matches old
configs/bytelm/bytelm_xs4_ctx256.py's intent but ported to run through qcute_v5_wordlm's shared
infra. Layer-count ablation pair against xs.py (n_layers=1) and xs2.py (n_layers=2).

uv run python -m qcute.qcute_v5_wordlm --config configs/v5_word/xs4.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_word_xs4
"""
from pathlib import Path

run_name = "v5_word_xs4"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 4
context_len = 256
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

qual_gen_bytes = 0
qual_prompt_bytes = 64

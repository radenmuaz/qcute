"""v5_word/xs2: same as xs.py but n_layers=2 -- layer-count ablation, matches old
configs/bytelm/bytelm_xs2_ctx256.py's intent (depth-matched against v5_stack/v5_concat decode
designs, whose own per-block decode adds extra effective depth beyond a plain n_layers=1
encoder) but ported to run through qcute_v5_wordlm's shared infra.

uv run python -m qcute.qcute_v5_wordlm --config configs/v5_word/xs2.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_word_xs2
"""
from pathlib import Path

run_name = "v5_word_xs2"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 2
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

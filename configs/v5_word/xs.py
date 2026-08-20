"""v5_word/xs: plain unconditional next-byte LM baseline (qcute.qcute_v5_wordlm.WordLM),
matched in scale (d_model=256, n_layers=1, context_len=256) to the v5_stack_* Ks=(1,) baselines
it's meant to be compared against -- same logs/run.jsonl format, so scripts/compare_runs.py can
overlay them directly. quant_type/decoder_type/Ks are required by the shared argparser but
unused (WordLM discards the quantized code entirely, no decode stage).

uv run python -m qcute.qcute_v5_wordlm --config configs/v5_word/xs.py

# plot after training:
uv run python scripts/plot_run.py logs/v5_word_xs
"""
from pathlib import Path

run_name = "v5_word_xs"
decoder_type = "stack"
Ks = (1,)
d_model = 256
n_layers = 1
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

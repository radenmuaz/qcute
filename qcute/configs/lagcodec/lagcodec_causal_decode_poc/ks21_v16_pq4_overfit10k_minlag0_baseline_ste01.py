"""v1_causal_decode_poc/ks21_v16_pq4_overfit10k_minlag0_baseline_ste01: same pairing as
ks21_v16_pq4_overfit10k_minlag0_baseline.py (own_code_min_lag=0, own-block reconstruction) but
with encoder_ste_p=0.1 -- the knob that gave a clear, monotonic exact-match improvement on the
standard own-code sequential decoder (docs/status.md's 2026-08-23 entry: base 9-10/50, ste_p=0.1
19-21/53, ste_p=1.0 27-31/50-51). Paired control for
ks21_v16_pq4_overfit10k_minlag1_causal_ste01.py -- isolates whether ste_p's benefit is specific
to own-block reconstruction or transfers to the causal (min_lag=1) retargeting too.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_causal_decode_poc/ks21_v16_pq4_overfit10k_minlag0_baseline_ste01.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_causal_decode_poc_ks21_v16_pq4_overfit10k_minlag0_baseline_ste01
"""
from pathlib import Path

run_name = "v1_causal_decode_poc_ks21_v16_pq4_overfit10k_minlag0_baseline_ste01"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
own_code_min_lag = 0
encoder_ste_p = 0.1
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64

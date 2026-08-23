"""v1_causal_decode_poc/ks21_v16_pq4_overfit10k_minlag1_causal: the POC itself (2026-08-23,
docs/maths.md Part 8/13). own_code_min_lag=1: track0 decodes block b's bytes conditioned ONLY
on strictly-earlier codes (c_{b-1} and older -- block b's OWN code is masked out), retargeting
decode from own-block reconstruction to genuine next-block prediction. Per Part 8's proof, this
makes bpb an exact chain-rule identity (like a plain AR LM), not merely an ELBO bound -- at the
cost of per-block fidelity, since a strictly-past code only forecasts block b rather than being
guaranteed sufficient to reconstruct it. Paired control:
ks21_v16_pq4_overfit10k_minlag0_baseline.py (own_code_min_lag=0, everything else identical).
Expect: higher train/val bpb and lower byte_acc than the baseline (this is not a regression --
it's the expected cost of getting a genuinely comparable, exact bpb instead of a bound).

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_causal_decode_poc/ks21_v16_pq4_overfit10k_minlag1_causal.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_causal_decode_poc_ks21_v16_pq4_overfit10k_minlag1_causal
"""
from pathlib import Path

run_name = "v1_causal_decode_poc_ks21_v16_pq4_overfit10k_minlag1_causal"
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
own_code_min_lag = 1
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

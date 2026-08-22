"""v1_stack_simplex/ks21_v256_pq1_overfit10k_kvlm_fresh: same setup as
ks21_v256_pq1_overfit10k.py (Ks=(2,1), the reliably-converging 2-level baseline) but
kv_lm_mode="fresh" -- level0's upper-track cross-attention K/V (level1's code) is now built by a
dedicated 1-layer causal self-attention+MLP pass over the embedded code sequence
(code_context_pass/KVContextLM, qcute_v1_decoder.py, chat 2026-08-22) instead of an isolated
per-position code embedding. Sanity/regression check: ks21 already converges cleanly without this,
so this run is mainly checking kv_lm doesn't destabilize the easy case, before judging its effect
on ks221 (the hard case) in the paired configs
ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum.py and
..._kvlm_fresh_curriculum.py.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v256_pq1_overfit10k_kvlm_fresh.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v256_pq1_overfit10k_kvlm_fresh
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v256_pq1_overfit10k_kvlm_fresh"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 256
pq_chunks = 1
kv_lm_mode = "fresh"
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64

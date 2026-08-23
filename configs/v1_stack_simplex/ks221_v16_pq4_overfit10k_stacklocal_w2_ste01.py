"""v1_stack_simplex/ks221_v16_pq4_overfit10k_stacklocal_w2_ste01: same as
ks221_v16_pq4_overfit10k_stacklocal_w2.py (stack_local, level0 cross-attn window = 2 level1
codes) plus encoder_ste_p=0.1 (additive, encoder_ste_skip_real=False default) -- the same knob
that gave a clear, monotonic exact-match improvement on the sequential StackDecoder ks221 ablation
(2026-08-23: base 9-10/50, ste_p=0.1 19-21/53, ste_p=1.0 27-31/50-51, all vs. train byte_acc
dropping only slightly). Tests whether that improvement transfers to the fully-parallel
block-local decode structure too.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack_local --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_stacklocal_w2_ste01.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_stacklocal_w2_ste01
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_stacklocal_w2_ste01"
decoder_type = "stack_local"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, 8, -1]),
    (-1, [32, -1]),
    -1,
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
kv_lm_mode = "copy"
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

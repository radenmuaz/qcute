"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window_conddepth1: same tiny-window stress test as
ks221_v256_pq1_overfit10k_window_conddepth1.py (Ks=(2,2,1), every non-top level's own
self-attention window forced to exactly its own K, cond_depth=1) but with vocab=16, pq_chunks=4
(4 independent 16-way softmaxes, 16-bit combinatorial code, close in total code width to the
original vocab=256 pq_chunks=1 setup's 8 bits) instead of the single 256-way softmax -- third
fallback after both the longer-steps (ks221_v256_pq1_overfit10k_window_long.py) and cond_depth=1
(ks221_v256_pq1_overfit10k_window_conddepth1.py) attempts failed to produce coherent real
generation (see docs/status.md's 2026-08-20 hard-convergence-queue entries). Queued alongside
ks21_v16_pq4_overfit10k_tinywindow_conddepth1.py as a sanity baseline.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq4_overfit10k_window_conddepth1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window_conddepth1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window_conddepth1"
decoder_type = "stack"
cond_depth = 1
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [2, -1, -1]),  # level0: own-block self-attn (track0)=K0=2, cross-attn to level1/level2 full
    (-1, [2, -1]),       # level1: own-block self-attn (track0)=K1=2, cross-attn to level2 full
    -1,                  # level2 (top): unchanged full
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 3000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 64  # check_gen_consistency/check_roundtrip_consistency/check_decode_modes all
# skip gracefully at n_levels==3 (StackDecoder's generation-fix work is n_levels==2-only so far,
# chat 2026-08-20) -- check_gen_consistency was missing that guard and crashed until fixed this
# same session; qualitative_generate's uncond/level_gen output still works and is worth seeing.

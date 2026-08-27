"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window4_relaxed: same window-relaxation isolation
test as ks221_v256_pq1_overfit10k_window4_relaxed.py (Ks=(2,2,1), pervasive cond_depth=-1, both
non-top levels' own self-attention windows relaxed from their own K to 2x their own K -- level0's
own-byte window: 2->4; level1's own-code window: 2->4 -- 3000 steps) but with vocab=16, pq_chunks=4
(16-bit combinatorial code) instead of the single 256-way softmax, combining the window-relaxation
variable with the PQ quant-type fallback that converged cleanly at ks21 (n_levels=2) but not yet
tested at ks221 with window relaxation specifically. See docs/status.md's 2026-08-20
hard-convergence-queue entries.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq4_overfit10k_window4_relaxed.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window4_relaxed
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window4_relaxed"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [4, -1, -1]),  # level0: own-block self-attn (track0) relaxed to 4 (2 blocks worth of
    # bytes, was 2), cross-attn to level1/level2 full
    (-1, [4, -1]),       # level1: own-block self-attn (track0) relaxed to 4 (2 blocks worth of
    # codes, was 2), cross-attn to level2 full
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

"""v1_stack_simplex/ks221_v256_pq1_overfit10k_window: tiny-window stress test (see
ks21_v256_pq1_overfit10k_tinywindow.py, docs/status.md's 2026-08-20 entries) generalized to
Ks=(2,2,1) (n_levels=3, one more non-top level than the ks21 configs). Every non-top level's own
byte/code-level self-attention window (its own-block reconstruction context, decode track0) is
forced to exactly its own K -- level0 window=K0=2 (one 2-byte block), level1 window=K1=2 (one
2-code block) -- while every cross-attention window onto a HIGHER level's code stays full/
unbounded (level0 still sees level1 and level2's codes fully; level1 still sees level2's code
fully; StackDecoder's default cond_depth=-1 keeps this pervasive). Tests whether the "own-code
decode doesn't need self-attention lookback to overfit" finding from Ks=(2,1) still holds with a
second non-top level added.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v256_pq1_overfit10k_window.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v256_pq1_overfit10k_window
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v256_pq1_overfit10k_window"
decoder_type = "stack"
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
vocab = 256
pq_chunks = 1
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

qual_gen_bytes = 64  # check_gen_consistency/check_roundtrip_consistency/check_decode_modes all
# skip gracefully at n_levels==3 (StackDecoder's generation-fix work is n_levels==2-only so far,
# chat 2026-08-20) -- check_gen_consistency was missing that guard and crashed until fixed this
# same session; qualitative_generate's uncond/level_gen output still works and is worth seeing.

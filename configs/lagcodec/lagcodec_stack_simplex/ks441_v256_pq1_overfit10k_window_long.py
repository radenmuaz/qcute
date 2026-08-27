"""v1_stack_simplex/ks441_v256_pq1_overfit10k_window_long: same as ks441_v256_pq1_overfit10k_window.py
(Ks=(4,4,1), every non-top level's own self-attention window forced to exactly its own K) but with
steps tripled to 3000 -- the 1000-step run reached decent train byte_acc (95.16%, teacher-forced)
but real generation still collapsed: level0_mode1's output was repetitive garble and BOTH
level1_gen and level2_gen degenerated to a single constant repeated token (see docs/status.md's
2026-08-20 hard-convergence-queue entry). Testing whether more steps lets the upper-level NTP
forecasts actually diversify instead of collapsing.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks441_v256_pq1_overfit10k_window_long.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks441_v256_pq1_overfit10k_window_long
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks441_v256_pq1_overfit10k_window_long"
decoder_type = "stack"
Ks = (4, 4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [4, -1, -1]),  # level0: own-block self-attn (track0)=K0=4, cross-attn to level1/level2 full
    (-1, [4, -1]),       # level1: own-block self-attn (track0)=K1=4, cross-attn to level2 full
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

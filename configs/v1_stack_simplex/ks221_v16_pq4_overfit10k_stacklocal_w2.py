"""v1_stack_simplex/ks221_v16_pq4_overfit10k_stacklocal_w2: same as
ks221_v16_pq4_overfit10k_stacklocal_w1.py (decoder_type="stack_local", block-diagonal same-level
decode at both level0 and level1) but level0's cross-attn window into level1's code widens to 2
codes (decode_windows[0][1] = 2*cum_K = 8 -- own block's level1 code PLUS one neighbor, chat
2026-08-23's "overlap is don't-care" extension). Tests whether the extra neighboring-code context
gives more coherent per-block decode than the window=1 case, per the same tunable-window knob
StackDecoder already exposes (no new mechanism, decode_windows only).

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack_local --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_stacklocal_w2.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_stacklocal_w2
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_stacklocal_w2"
decoder_type = "stack_local"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, 8, -1]),   # level0: track1 (level1 code) = 8 = 2 codes (own block + 1 neighbor)
    (-1, [32, -1]),
    -1,
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
kv_lm_mode = "copy"
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

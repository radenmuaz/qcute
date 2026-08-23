"""v1_stack_simplex/ks221_v16_pq4_overfit10k_stacklocal_w1: same base setup as
ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel.py (Ks=(2,2,1), vocab=16
pq_chunks=4, kv_lm_mode="fresh") but decoder_type="stack_local" (StackDecoderLocal,
block_local_track0_decode) instead of "stack" -- every block at level0 AND level1 decodes its
OWN bytes/code fully independently of every other block at the SAME level (zero cross-block
same-level visibility, block-diagonal by construction, all blocks one parallel batched call --
see StackDecoderLocal's docstring, qcute_v1_decoder.py). The only remaining cross-block
information channel is cross-attention to the level ABOVE's code, which stays tunable via
decode_windows same as StackDecoder.

This config: level0's cross-attn window into level1's code is set to exactly 1 code
(decode_windows[0][1] = cum_K = Ks[0]*Ks[1] = 4 -- the SIMPLEST case, own block's level1 code
only, chat 2026-08-23). Since the topmost level (level2) is now hard-excluded from all
conditioning (StackDecoder.__init__, 2026-08-23), level1 itself has ZERO upper tracks left --
its decode is therefore ALREADY fully parallel/block-local with no window question at all.
Confirms whether the fully-parallel-given-code decode structure trains/overfits at all before
widening the window or adding encoder_ste_p (see ..._w2.py, ..._w2_ste01.py).

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack_local --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_stacklocal_w1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_stacklocal_w1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_stacklocal_w1"
decoder_type = "stack_local"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, 4, -1]),   # level0: track0 dw (32) ignored by stack_local; track1 (level1 code) = 4 = 1 code
    (-1, [32, -1]),      # level1: track0 dw (32) ignored; track1 unused (level2 hard-excluded)
    -1,
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
kv_lm_mode = "fresh"
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

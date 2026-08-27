"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum2: fixes
ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum.py's curriculum, which used a scalar
active_srcs_mode=2 -- that only drops level2 from level0's upper tracks; level1 (only 1 upper
track: level2) keeps seeing level2 regardless, since a scalar cap can't drop a level's OWN single
nearest upper track without also zeroing level0's (see chat 2026-08-21). active_srcs_mode is
now a per-level tuple (2, 1, None): level0 keeps only level1 (drops level2), level1 drops its
only upper track (level2) too -- for step < active_srcs_until_step, the model's decode genuinely has NO
path from level2's code into anything, matching what a real standalone Ks=(2,1) ks21 model would
see. From active_srcs_until_step onward, max_srcs reverts to None (full ks221, uniform for every level).
Same base setup otherwise: window16_relaxed, vocab=16 pq_chunks=4, scheduled_sampling_p=1.0,
detach_ss_sample=False, uncertainty_weighting=False.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum2.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum2
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum2"
decoder_type = "stack"
Ks = (2, 2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (
    (-1, [32, -1, -1]),
    (-1, [32, -1]),
    -1,
)
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
scheduled_sampling_p = 1.0
detach_ss_sample = False
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000
active_srcs_mode = (2, 1, None)
active_srcs_until_step = steps // 2

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

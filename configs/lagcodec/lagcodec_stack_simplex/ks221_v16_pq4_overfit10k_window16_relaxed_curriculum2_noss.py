"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss: isolation test for
ks221_v16_pq4_overfit10k_window16_relaxed_ss1_curriculum2.py (the first run to produce coherent
real ks221 generation, see docs/status.md's 2026-08-21/22 entry) -- identical setup (Ks=(2,2,1),
window16_relaxed, vocab=16 pq_chunks=4, active_srcs_mode=(2,1,None) for the first half of
training then None/full for the second half) but scheduled_sampling_p=0 (no scheduled sampling at
all, vs curriculum2's ss=1.0). curriculum2's base setup already had scheduled_sampling_p=1.0
enabled from an earlier (failed on its own) lever-sweep -- this isolates whether the
active_srcs_mode/active_srcs_until_step mechanism alone is sufficient for the coherent-generation
result, or whether it needed full-strength scheduled sampling alongside it.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss"
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
scheduled_sampling_p = 0.0
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

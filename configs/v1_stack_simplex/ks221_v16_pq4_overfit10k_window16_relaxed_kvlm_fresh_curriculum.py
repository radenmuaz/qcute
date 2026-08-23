"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_curriculum: same as
ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum.py (kv_lm_mode="fresh" on top of
the best confirmed non-curriculum recipe) but ALSO applies the validated curriculum
(active_srcs_mode=(2,1,None), active_srcs_until_step=steps//2 -- the recipe that first produced
coherent ks221 generation, see ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss.py and
docs/status.md's 2026-08-21/22 entry). Run after the no-curriculum kv_lm variant to see whether
kv_lm alone already suffices (in which case this should look similar) or whether the curriculum is
still doing separate, additive work on top of kv_lm.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_curriculum.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_curriculum
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_curriculum"
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
kv_lm_mode = "fresh"
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

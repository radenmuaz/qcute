"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum: same base
setup as ks221_v16_pq4_overfit10k_window16_relaxed_curriculum2_noss.py (Ks=(2,2,1), window16_relaxed,
vocab=16 pq_chunks=4, no scheduled sampling, no uncertainty weighting -- the best confirmed recipe
so far, see docs/status.md's 2026-08-21/22 entry) but NO curriculum (curriculum_max_srcs unset) and
kv_lm_mode="fresh": every upper-track cross-attention K/V (level0's tracks to level1's/level2's
code, level1's track to level2's code) now goes through a dedicated 1-layer causal self-attention
pass (code_context_pass/KVContextLM, chat 2026-08-22) before being used as K/V, instead of an
isolated per-position code embedding. Tests whether kv_lm alone (no curriculum) is enough to avoid
the repetitive-collapse failure that every non-curriculum ks221 lever tried this session failed to
fix. Paired with ..._kvlm_fresh_curriculum.py (same but WITH the curriculum) to see whether kv_lm
and the curriculum are complementary, redundant, or whether kv_lm alone suffices.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum"
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
kv_lm_mode = "copy"
scheduled_sampling_p = 0.0
detach_ss_sample = False
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

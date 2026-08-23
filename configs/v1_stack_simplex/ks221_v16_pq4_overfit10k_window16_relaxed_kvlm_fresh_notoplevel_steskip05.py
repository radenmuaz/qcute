"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_steskip05: same
as ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_ste01.py but
encoder_ste_p=0.5, encoder_ste_skip_real=True (skip/substitution mode, the old scheduled_sampling_p
behavior) instead of additive -- half the forward passes, every non-top level's decode is fed the
level-above's own self-sampled code INSTEAD OF the real one (mutually exclusive, not additive), the
other half real ground-truth as usual. Part of the same 4-way ablation
(encoder_ste_p in {0, 0.1, 1.0} additive, plus this 0.5 skip_real variant) on the n_levels=3 case --
tests whether the skip/substitution mode's earlier instability on ks21 (n_levels=2) replicates here.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_steskip05.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_steskip05
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_steskip05"
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
encoder_ste_p = 0.5
encoder_ste_skip_real = True
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000
# curriculum_max_srcs/curriculum_step (notoplevel exclusion) removed 2026-08-23 -- now baked
# into StackDecoder.__init__ unconditionally, no curriculum needed (see qcute_v1_decoder.py's
# StackDecoder docstring).

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

"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_ste01: same as
ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel.py (Ks=(2,2,1), vocab=16/pq_chunks=4,
kv_lm_mode="fresh", notoplevel: curriculum_max_srcs=(2,1,None) held for the WHOLE run -- excludes
level2, the topmost level, from level0/level1's decode conditioning) but encoder_ste_p=0.1,
encoder_ste_skip_real=False (additive mode, default): 1 in 10 forward passes runs a SEPARATE second
decode pass fed every non-top level's own self-sampled code, added unweighted on top of the always-
present real-code decode loss -- giving level1's/level2's own NTP heads direct gradient signal for
how well decode reconstructs from their own samples. Part of a 4-way ablation
(encoder_ste_p in {0 (this file's base), 0.1, 1.0} additive, plus 0.5 skip_real) on the harder
n_levels=3 case, following up on the ks21 (n_levels=2) comparison in docs/status.md.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_ste01.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_ste01
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel_ste01"
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
encoder_ste_p = 0.1
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 3000
# curriculum_max_srcs/curriculum_step (notoplevel exclusion) removed 2026-08-23 -- now baked
# into StackDecoder.__init__ unconditionally, no curriculum needed (see qcute_lagcodec_decoder.py's
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

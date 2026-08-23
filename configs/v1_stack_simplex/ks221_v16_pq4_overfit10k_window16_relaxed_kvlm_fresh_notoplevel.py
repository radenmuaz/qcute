"""v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel: same base
setup as ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_nocurriculum.py (Ks=(2,2,1),
window16_relaxed, vocab=16 pq_chunks=4, kv_lm_mode="fresh", no scheduled sampling -- the current
best confirmed ks221 recipe, see docs/status.md's 2026-08-21/22 entry) but forces every non-top
level's decode to train WITHOUT any conditioning on level2 (the topmost level)'s code -- via
curriculum_max_srcs=(2, 1, None) (level0: own+level1 only; level1: own code only) held active for
the WHOLE run (curriculum_step set past total steps -- not a phased curriculum, a permanent
topology cap, same mechanism as the earlier curriculum2/curriculum2_noss runs but with the DIFFERENT
tuple that specifically excludes only the top level, not "half the levels for half of training").

Motivation: level2 is the topmost level here -- nothing generatively models its own code from above,
so at real generation time it's only ever the product of its own free-running self-NTP, never
grounded/verified by a further level. Conditioning level0/level1's decode on that ungrounded code is
the exposure-bias-prone substitution from docs/status.md's "real qcute_v1 vs qcute_zero
differentiator" section -- this run tests whether dropping it (while KEEPING level0's conditioning
on level1, which IS grounded by level2 one level up) changes anything vs. the kvlm_fresh_nocurriculum
baseline (best val bpb 1.93 so far).

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks221_v16_pq4_overfit10k_window16_relaxed_kvlm_fresh_notoplevel"
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

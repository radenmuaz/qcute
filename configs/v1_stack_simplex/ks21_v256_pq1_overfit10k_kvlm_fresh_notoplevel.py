"""v1_stack_simplex/ks21_v256_pq1_overfit10k_kvlm_fresh_notoplevel: same base setup as
ks21_v256_pq1_overfit10k_kvlm_fresh.py (Ks=(2,1), kv_lm_mode="fresh") but forces level0's decode to
train WITHOUT any conditioning on level1's code at all -- via curriculum_max_srcs=(1, None) (max_srcs
convention: 1 = own code only, no upper tracks) held active for the WHOLE run (curriculum_step set
past total steps, so this is not an actual curriculum/phased schedule, just a permanent topology cap
-- see docs/status.md's 2026-08-22 "exclude the topmost level's code" entry and chat that day).

Motivation: level1 is the topmost level in this Ks=(2,1) config -- nothing generatively models its
own code from above (no level2), so at real generation time level1's code is only ever the product
of its own free-running self-NTP, never grounded/verified by a further level. Conditioning level0's
decode on that ungrounded code is exactly the exposure-bias-prone substitution discussed in
docs/status.md's "real qcute_v1 vs qcute_zero differentiator" section -- this run tests whether
dropping it entirely (own-code-only decode, no hierarchy benefit for level0 at all) changes anything
vs. the existing kvlm_fresh baseline.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v256_pq1_overfit10k_kvlm_fresh_notoplevel.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v256_pq1_overfit10k_kvlm_fresh_notoplevel
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v256_pq1_overfit10k_kvlm_fresh_notoplevel"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 256
pq_chunks = 1
kv_lm_mode = "fresh"
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 1000
curriculum_max_srcs = (1, None)   # own code only for level0, no conditioning on level1 (the top level)
curriculum_step = steps + 1       # active for the WHOLE run -- not a phased curriculum

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

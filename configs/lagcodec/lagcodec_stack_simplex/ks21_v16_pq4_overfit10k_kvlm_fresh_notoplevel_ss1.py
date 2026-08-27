"""v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_ss1: same base setup as
ks21_v256_pq1_overfit10k_kvlm_fresh.py (Ks=(2,1), kv_lm_mode="fresh") but vocab=16/pq_chunks=4
instead of vocab=256/pq_chunks=1 (v256pq1 abandoned as a base -- consistently the weaker
quant-structure choice throughout this investigation, see docs/status.md's quant-structure sweep),
plus encoder_ste_p=1.0 with encoder_ste_skip_real=True (STE-connected, detach_ss_sample=False
default) -- the old scheduled_sampling_p behavior, unified into encoder_ste_p 2026-08-23 (see
qcute_lagcodec_common.py's Config.encoder_ste_skip_real docstring): every forward pass, level0's decode is
fed level1's OWN sampled code prediction instead of the ground-truth code (mutually exclusive with
the real-code pass, not additive), closing the train/generation exposure-bias gap on the code that
feeds decode. Real test: does the
free-rollout dependency confirmed in docs/status.md's 2026-08-22/23 entries (level0's own code is
always sourced via level1's generative forecast during real generation, regardless of any
max_srcs/cond_depth capping) actually improve when decode is trained end-to-end against that same
self-sampled code every step, rather than only at generation time.

ALSO notoplevel (curriculum_max_srcs=(1, None), curriculum_step past total steps -- active for the
WHOLE run, not a phased curriculum): level0's decode cross-attention never conditions on level1's
code at all, own-code-only, matching the earlier ks21_v256_pq1_overfit10k_kvlm_fresh_notoplevel.py
experiment's topology but now combined with ss1.0.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_ss1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_ss1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_ss1"
decoder_type = "stack"
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 16
pq_chunks = 4
kv_lm_mode = "copy"
encoder_ste_p = 1.0
encoder_ste_skip_real = True
detach_ss_sample = False
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 1000
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

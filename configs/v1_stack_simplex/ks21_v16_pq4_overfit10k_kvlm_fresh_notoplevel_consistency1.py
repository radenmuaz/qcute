"""v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency1: same base setup as
ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_ss1.py (Ks=(2,1), vocab=16/pq_chunks=4,
kv_lm_mode="fresh", notoplevel: curriculum_max_srcs=(1, None) held for the WHOLE run -- level0's
decode cross-attention never conditions on level1's code at all, own-code-only) but
encoder_ste_p=1.0, encoder_ste_skip_real=False default (additive mode; renamed from consistency_p
2026-08-23, see qcute_v1_common.py's Config.encoder_ste_p/encoder_ste_skip_real docstrings) --
every forward pass, decode is run a SECOND time fed level1's own self-sampled code (same
sample_next() mechanism the skip-mode/formerly-scheduled_sampling_p path uses), and that
reconstruction loss is added UNWEIGHTED and IN ADDITION to (never replacing) the real-code decode
loss, giving the level-above's own NTP head direct gradient signal for how well decode can
reconstruct from ITS OWN sample (docs/status.md's "code-level consistency training" discipline
#2/#3). Paired with the ss1 config (encoder_ste_skip_real=True) to compare substitution against
addition as two different fixes for the same free-rollout dependency, both under the notoplevel
topology.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency1"
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
kv_lm_mode = "fresh"
encoder_ste_p = 1.0
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

steps = 1000
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

"""v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency05: same as
ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency1.py but byte_consistency_p=0.5
instead of 1.0 -- half the forward passes run the extra whole-model self-feeding pass, half don't.
Checks whether p=1.0's always-on extra pass is too aggressive (same open question raised for
encoder_ste_p/scheduled_sampling_p earlier, docs/status.md), now for the whole-model byte-space
mechanism instead of the code-level one.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency05.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency05
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency05"
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
byte_consistency_p = 0.5
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

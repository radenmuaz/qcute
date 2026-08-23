"""v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency01: same as
ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency1.py (Ks=(2,1), vocab=16/pq_chunks=4,
kv_lm_mode="fresh", notoplevel: curriculum_max_srcs=(1, None) held for the WHOLE run) but
encoder_ste_p=0.1 instead of 1.0 (additive mode, encoder_ste_skip_real default False; renamed from
consistency_p 2026-08-23) -- only 1 in 10 forward passes runs the extra self-sampled-code decode
pass, vs. every step. Checks whether p=1.0's always-on extra loss is too aggressive (same open
question the skip-mode/formerly-scheduled_sampling_p p=1.0/0.5/0.1 sweep raised earlier,
docs/status.md), now for the additive mechanism instead of the substitution one.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency01.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency01
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency01"
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
encoder_ste_p = 0.1
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

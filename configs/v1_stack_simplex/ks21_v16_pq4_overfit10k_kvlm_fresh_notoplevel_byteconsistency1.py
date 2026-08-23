"""v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency1: same base setup
as ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel.py (Ks=(2,1), vocab=16/pq_chunks=4,
kv_lm_mode="fresh", notoplevel: curriculum_max_srcs=(1, None) held for the WHOLE run) but
byte_consistency_p=1.0: every forward pass, level0's own byte-level reconstruction logits are
argmax'd, detached, and fed through the WHOLE model again (self-supervised as always -- the second
pass reconstructs its OWN input), testing whole-model idempotence/stability under one round of
self-feeding -- a genuinely different, bigger mechanism than encoder_ste_p's code-level-only swap
(new Config.byte_consistency_p, qcute_v1.py, 2026-08-23). Real cost: this doubles the whole
forward pass (every level's encoder + decode), not just decode_level, whenever it fires.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency1.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency1
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_byteconsistency1"
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
byte_consistency_p = 1.0
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

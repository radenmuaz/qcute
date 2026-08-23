"""v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel: same base setup as
ks21_v256_pq1_overfit10k_kvlm_fresh_notoplevel.py (Ks=(2,1), kv_lm_mode="fresh", notoplevel:
curriculum_max_srcs=(1, None) held for the WHOLE run -- level0's decode cross-attention never
conditions on level1's code at all, own-code-only) but vocab=16/pq_chunks=4 instead of
vocab=256/pq_chunks=1 (v256pq1 abandoned as a base -- consistently the weaker quant-structure
choice throughout this investigation, see docs/status.md's quant-structure sweep). Plain baseline
(no scheduled_sampling_p, no consistency_p) -- paired with
ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_ss1.py and
ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel_consistency1.py to isolate what each lever adds on
top of this same v16pq4/notoplevel base.

uv run python -m qcute.qcute_v1.qcute_v1 --decoder_type stack --config configs/v1_stack_simplex/ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v16_pq4_overfit10k_kvlm_fresh_notoplevel"
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

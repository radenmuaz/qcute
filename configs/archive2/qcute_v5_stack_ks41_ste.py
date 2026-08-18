"""qcute_v5_stack_ks41_ste: configs/qcute_v5_concat_modes_1_ste.py cloned onto qcute.qcute_v5_stack
-- same Ks=(4,1)/d_model/context_len/attn_window/code_sample_mode="ste"/schedule, so results are
directly comparable to that concat ablation. qcute_v5_stack.py has no multi_mode_impl flag (unlike
concat's fork) since staged cross-attention decode already trains encode + decode-self (stage 0,
self_merged_stage) + every decode-cross-attn stage up through the deepest/full stage "for free" as
a byproduct of the sequential per-track chain -- decode_total (deepest stage) and
decode_stage_extra_total (self + any intermediate stages, pooled) are both already part of the
loss, see RefineLM._run/.forward. attn_window=-1 (unbounded, matches the concat ablation's choice).

uv run python -m qcute.qcute_v5_stack --config configs/qcute_v5_stack_ks41_ste.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_stack_ks41_ste
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_sample_mode = "ste"

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
eval_batches = 20

qual_gen_bytes = 128
qual_prompt_bytes = 64

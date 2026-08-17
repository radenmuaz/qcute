"""qcute_v5_concat_modes_1: configs/qcute_v5_concat_modes_soft.py cloned to Ks=(4,1) (n_levels=2)
-- unlike the Ks=(1,) soft config, this actually exercises multi_mode_impl="single_pass":
level0 has T=2 tracks (self K=4, +1 K=4 since Ks[1]=1 doesn't widen the span), picking up one
shallower mode (self-only); level1 stays T=1 (no extra mode). code_sample_mode="soft" (plain
Gumbel-Softmax relaxation, no hard forward) carried over unchanged. attn_window=-1 (unbounded) --
originally tried (256,256), but any FINITE window triggers the chunked/banded attention path once
codes are folded into the merged buffer (buffer length exceeds the window), and
multi_mode_impl="single_pass" only supports the dense path (asserts loudly rather than silently
computing something wrong) -- -1 stays dense and is equivalent anyway since context_len=256.

uv run python -m qcute.qcute_v5_concat_modes --config configs/qcute_v5_concat_modes_1.py

# plot after training:
uv run python scripts/plot_run.py logs/qcute_v5_concat_modes_1
"""
from pathlib import Path

Ks = (4, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = (8,64)
code_sample_mode = "soft"
multi_mode_impl = "single_pass"
# gumbel_tau = 1.0

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 200
eval_every = 2000
# eval_every = 100
eval_batches = 20

qual_gen_bytes = 128
qual_prompt_bytes = 64

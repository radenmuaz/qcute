"""qcute.qcutelm_vlt6 config: quant_type="none" — no quantization/codebook
at all, fully continuous code (identity passthrough at the bottleneck).
Diagnostic ablation: isolates whether the discrete BSQ/FSQ/iFSQ bottleneck
itself is costing real convergence speed, by comparing directly against
qcutelm_vlt6_ifsq_2xflops_leanparams.py (identical architecture/schedule,
only quant_type differs).

Caveat (see session notes): even if this converges faster, it isn't a
drop-in replacement for the project's actual generative goal — codepred
here is trained by pure backprop with no distributional loss of its own,
so it will tend toward regression-to-the-mean rather than a genuinely
samplable next-code distribution. This run is purely diagnostic.

    uv run python -m qcute.qcutelm_vlt6 --config configs/qcutelm_vlt6_none_2xflops.py
"""
from pathlib import Path

K = 4
context_len = 1024
attn_window = 64
dq = 6
quant_type = "none"

d_model = 96
n_heads = 4
n_layers = 2
mlp_mult = 4
code_net_layers = 0

lm_d_model = 256
lm_n_heads = 4
lm_n_layers = 3
lm_mlp_mult = 4

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 8000
batch_size = 16
lr_peak = 6e-4
weight_decay = 1e-5
warmup_steps = 500
cosine_decay = False
constant_steps = 100
eval_every = 100
eval_batches = 20

gen_every = 200
gen_prompt_len = 64
gen_new_bytes = 64

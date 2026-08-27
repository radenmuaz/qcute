"""qcute.bytelm_tpu config: v4-8 multichip (4-way data-parallel) full training run of the
LLaMA-style arch (RMSNorm, SwiGLU, bias-free) at ~67M params (d_model=512, n_layers=16,
n_heads=8, mlp_mult=4 -- see PRESETS["d512x16"]), context=8192, use_flash_attention=True,
no_zero_kv_sink.

Runs on tpu5 (v4-8, us-central2-b), set up fresh with the NIGHTLY torch/torch_xla build
(torch==2.10.0.dev0, libtpu==0.0.46) per docs/tpu_setup.md's "Optional: nightly build"
section -- tpu4's bytelm_tpu_v4-8_d512_16L_multichip.py runs the stable pin instead, which
cannot do flash-attention at all (see that config's own docstring for the full story of why).
This is the first real test of --multichip + --use_flash_attention together, a combination
flagged as untested in both CLAUDE.md and docs/tpu_setup.md before this run.

batch_size=4 -- smart-guess per-process batch, NOT bench-verified on this node. Flash-attention's
O(T) (not O(T^2)) memory means this could likely go higher than the plain-SDPA tpu4 config's
batch=4, but kept the same here as a conservative first attempt given the untested
multichip+flash+nightly combination -- bump later once this is confirmed stable.
torch.compile left OFF (--no_torch_compile) for the same reason. global_batch = 4*4 = 16.

Same retuned hyperparameters as bytelm_tpu_v6e1_d512_16L_lr1e4.py (lr_peak=1e-4, cosine_decay,
weight_decay=0.01, grad_clip=10.0).

    TPU_VISIBLE_CHIPS=0,1,2,3 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d512_16L_multichip_flash.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d512_16L_multichip_flash
"""
from pathlib import Path

preset = "d512x16"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
multichip = True
no_torch_compile = True
steps = 100000
batch_size = 4
lr_peak = 1e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 10.0
weight_decay = 0.01
log_every = 100
eval_every = 9000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1

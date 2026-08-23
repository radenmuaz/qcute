"""qcute.bytelm_tpu config: v4-8 multichip (4-way data-parallel) full training run of the
LLaMA-style arch (RMSNorm, SwiGLU, bias-free) at ~67M params (d_model=512, n_layers=16,
n_heads=8, mlp_mult=4 -- see PRESETS["d512x16"]), context=8192, plain masked SDPA (no
flash-attention -- see below), no_zero_kv_sink.

use_flash_attention=False here: tpu4 runs the STABLE torch_xla==2.9.0 pin
(libtpu==0.0.21), which per this repo's own findings does not support the Pallas
flash-attention kernel at all (needs a nightly build, libtpu>=0.0.44 -- see
docs/bytelm_tpu_setup.md's "Optional: nightly build" section). A first attempt at
--use_flash_attention here (2026-08-23) additionally installed jax[tpu] to satisfy the
kernel's internal jax dependency, which silently upgraded libtpu to 0.0.46 as a side
effect and broke --multichip's grpc-based multi-process TPU coordination entirely
(RuntimeError: "Failed to establish SliceBuilder grpc channel to localhost:8476") --
libtpu was reverted to 0.0.21 to fix that. Getting real flash-attention on this node
would require installing nightly torch_xla (as was done on tpu1), and
--multichip + --use_flash_attention together is explicitly untested even there.

batch_size=4 is a smart-guess per-process batch, NOT bench-verified on this node (tpu4, v4-8,
us-central2-b) -- carried over from the v6e-1 single-chip lr1e4 config's own batch=4, which *was*
bench-verified there (compiled batch=8 OOMed, batch=4 fine). Deliberately conservative / does not
attempt to saturate v4-8 memory -- per user instruction to skip benchmarking and launch directly.
global_batch = batch_size * world_size = 4 * 4 = 16. torch.compile left OFF here
(--no_torch_compile) since compile has not been verified in combination with --multichip on this
node -- reduces risk of an untested failure mode on a long unattended run.

Same retuned hyperparameters as bytelm_tpu_v6e1_d512_16L_lr1e4.py (lr_peak=1e-4, cosine_decay,
weight_decay=0.01, grad_clip=10.0) after that run showed overfitting with lr_peak=3e-4.

    TPU_VISIBLE_CHIPS=0,1,2,3 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d512_16L_multichip.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d512_16L_multichip
"""
from pathlib import Path

preset = "d512x16"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = False
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

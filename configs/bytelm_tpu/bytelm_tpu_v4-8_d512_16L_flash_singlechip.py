"""qcute.bytelm_tpu config: single-chip (v4-8 chip 0, via TPU_VISIBLE_CHIPS=0) full training run
of the LLaMA-style arch (RMSNorm, SwiGLU, bias-free) at ~67M params (d_model=512, n_layers=16,
n_heads=8, mlp_mult=4 -- see PRESETS["d512x16"]), context=8192, use_flash_attention=True,
no_zero_kv_sink. Runs on tpu5 (v4-8, us-central2-b), set up with the NIGHTLY torch/torch_xla
build (torch==2.10.0.dev0, libtpu==0.0.46) since flash-attention needs libtpu>=0.0.44.

Single-chip, not --multichip: bytelm_tpu_v4-8_d512_16L_multichip_flash.py (same node) confirmed
--multichip + --use_flash_attention hangs (all 4 workers' CPU time flat across repeated
snapshots, 5-step smoke test) -- see docs/tpu_setup.md's "Optional: multiple TPU chips on
one host" section for the full writeup. Standalone flash-attention works fine on this node/build;
it's specifically the multichip+flash combination that's broken on both the stable and nightly
pins tried so far. This config uses one chip alone instead, to get flash-attention actually
exercised in a real run.

batch_size=8: bench-probed on this node (2026-08-23) with a 5-step smoke sweep -- bs=8 completed
cleanly (first_step_compile_s=16.0), bs=16/24/32 all OOM'd (53.7G / 82.3G / 112.2G required vs.
31.75G HBM available, scaling ~linearly with batch -- consistent with flash-attention's O(T) not
O(T^2) memory, still large at this context/model size). Not narrowed further between 8 and 16.

Same retuned hyperparameters as bytelm_tpu_v6e1_d512_16L_lr1e4.py (lr_peak=1e-4, cosine_decay,
weight_decay=0.01, grad_clip=10.0).

    TPU_VISIBLE_CHIPS=0 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v4-8_d512_16L_flash_singlechip.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v4-8_d512_16L_flash_singlechip
"""
from pathlib import Path

preset = "d512x16"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
no_torch_compile = True
steps = 100000
batch_size = 8
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

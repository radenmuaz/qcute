"""qcute.bytelm_tpu config: single-chip (v6e-1) full training run of the new LLaMA-style arch
(RMSNorm, SwiGLU, bias-free) at ~67M params (d_model=512, n_layers=16, n_heads=8, mlp_mult=4 --
see PRESETS["d512x16"]), context=8192, torch.compile (openxla backend) ON.

Sizing, from live probes on this node (2026-08-23):
  - zero_kv_sink (default on) forces the O(T^2)-memory explicit-mask SDPA path even with
    use_flash_attention set (Pallas kernel can't combine the two) -- disabled here
    (no_zero_kv_sink=True) so flash-attention's memory savings actually apply.
  - plain fp32 was the first real memory bottleneck (bf16 autocast added, ~halves activation
    memory) before zero_kv_sink turned out to be the bigger one.
  - torch.compile (openxla backend): confirmed working, ~25% steady-state speedup over
    uncompiled at batch=4 (2.5 vs 2.0 it/s) but roughly halves max batch before OOM (batch=8
    OOMs compiled, fine uncompiled) -- picked compiled + batch=4 per user call, prioritizing
    per-step speed over batch-size headroom.
  - steps=100000 sized for ~11.1h at the measured 2.5 it/s (40000s), leaving ~1h margin under a
    12h budget for eval overhead -- ~39 epochs over enwik8's ~90M-byte train split. Re-tune
    --steps once real throughput at this exact config/step is confirmed early in the run (watch
    actual elapsed-time/it-rate, don't trust this estimate blindly -- see CLAUDE.md).

This run (lr_peak=3e-4) showed clear overfitting and was separately killed by an unrelated infra
issue before a stop-for-overfitting decision was made (tmux server died mid-run at step ~51000,
no traceback in qcute's own log -- see docs/tpu_direct_ssh.md). val_bpb bottomed at step 18000
(1.220, preserved at ~/qcute/checkpoints/bytelm_tpu_v6e1_d512_16L_saturate/best.pt on the node)
then rose for 3 consecutive evals (27000: 1.232, 36000: 1.275, 45000: 1.348, worse than the very
first eval). Retuned with a lower lr_peak in bytelm_tpu_v6e1_d512_16L_lr1e4.py (distinct
run_name, doesn't touch this run's checkpoint) -- that's the config to use going forward, this
one is kept only as the as-run historical record.

    TPU_VISIBLE_CHIPS=0 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v6e1_d512_16L_saturate.py --device xla --no_zero_kv_sink

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v6e1_d512_16L_saturate
"""
from pathlib import Path

preset = "d512x16"
data = Path("datasets/enwik8.gz")
val_frac = 0.05
test_frac = 0.05
context = 8192
use_flash_attention = True
no_zero_kv_sink = True
steps = 100000
batch_size = 4
lr_peak = 3e-4
warmup_steps = 1000
cosine_decay = True
constant_steps = 5000
grad_clip = 1.0
log_every = 100
eval_every = 9000
full_val_eval = True
eval_batches = 20
save_every_n_evals = 1

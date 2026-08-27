"""qcute.bytelm_tpu config: single-chip (v6e-1) full training run of the new LLaMA-style arch
(RMSNorm, SwiGLU, bias-free) at ~67M params (d_model=512, n_layers=16, n_heads=8, mlp_mult=4 --
see PRESETS["d512x16"]), context=8192, torch.compile (openxla backend) ON.

Sizing, from live probes on this node (2026-08-23), carried over from the original
bytelm_tpu_v6e1_d512_16L_saturate.py:
  - zero_kv_sink DISABLED here (context=8192, no_zero_kv_sink=True) -- see
    docs/tpu_setup.md's "zero_kv_sink + flash-attention: investigation" section for the
    full story. Short version: a "square" fix (pad Q by one dummy row so q_len==kv_len, verified
    numerically correct) makes the sink combine with flash-attention without OOMing, but costs
    ~25x steady-state throughput (10s/it vs 0.4s/it) on torch_xla -- confirmed not a
    torch.compile artifact (same slowdown without it) and not fixable by reusing a buffer instead
    of reallocating the zero tensor each call (no effect). A from-scratch JAX reimplementation
    (qcute/bytelm_jax.py) narrowed the gap to ~3s/it, but a controlled test (sink on vs off,
    fp32 vs bf16, all four combinations) showed the sink made **no measurable difference** in
    JAX either (0.330-0.350 it/s across all four) -- so JAX's own ~3s/it floor has some other,
    unidentified cause unrelated to the sink, and the earlier "JAX is 3.3x faster" read was
    wrong/confounded. Bottom line: no version of zero_kv_sink-with-flash-attention beats plain
    flash-attention without the sink (0.4s/it on torch_xla) at this scale, so it's off here.
  - plain fp32 was the first real memory bottleneck (bf16 autocast added, ~halves activation
    memory) before the zero_kv_sink/flash investigation above.
  - torch.compile (openxla backend): confirmed working, ~25% steady-state speedup over
    uncompiled at batch=4 (2.5 vs 2.0 it/s) but roughly halves max batch before OOM (batch=8
    OOMs compiled, fine uncompiled) -- picked compiled + batch=4 per user call, prioritizing
    per-step speed over batch-size headroom.
  - steps=100000 sized for ~11.1h at the measured 2.5 it/s (40000s) -- ~39 epochs over enwik8's
    ~90M-byte train split.

Retuned 2026-08-23 (this file, distinct run_name from the original so its checkpoint isn't
overwritten) after the original lr_peak=3e-4 run showed clear overfitting: val_bpb bottomed at
step 18000 (1.220) then rose for 3 consecutive evals (27000: 1.232, 36000: 1.275, 45000: 1.348,
worse than the very first eval) -- lr_peak dropped 3e-4 -> 1e-4 (cosine_decay kept on),
weight_decay set to 0.01 (default was 0.1; tried 0.3 and 0.2 first before settling here, per
user calls), grad_clip
raised 1.0 -> 10.0 (looser, per user call -- not verified against this run's own gradient-norm
distribution, just a deliberate loosening now that lr itself is smaller). That original run was separately
killed by an unrelated infra issue (tmux server died mid-run at step ~51000, no traceback in
qcute's own log -- coincided with a `gcloud ... ssh --command` reconnect call, whose "preparing
node" step may kill lingering user tmux sessions; see docs/tpu_direct_ssh.md) before a
stop-for-overfitting decision was made, but the retune here addresses the overfitting regardless.
Original run's best.pt (step 18000, val_bpb 1.220) is preserved at
~/qcute/checkpoints/bytelm_tpu_v6e1_d512_16L_saturate/best.pt on the node, untouched by this run.

    TPU_VISIBLE_CHIPS=0 uv run python -m qcute.bytelm_tpu --config configs/bytelm_tpu/bytelm_tpu_v6e1_d512_16L_lr1e4.py --device xla

    # plot after/during training:
    uv run python scripts/plot_run.py logs/bytelm_tpu_v6e1_d512_16L_lr1e4
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

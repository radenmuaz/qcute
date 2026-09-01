"""gpt2_jax.train_gpt config: Cable's own exact defaults, "medium" model (n_layer=24, n_head=16,
n_embd=1024, ~354M params -- Cable's README notes ~1.5GB checkpoint for this shape), rope
pos_method, 1 epoch over FineWeb-Edu-10B at Cable's own total_batch_size=524288 (2**19 tokens),
lr_peak=6e-4, batch_size=16/device (Cable's own MICRO_BATCH_SIZES["medium"]).

Paper trained on 8x H100 (per-GPU micro batch B, total_batch_size=524288 via grad accumulation --
see gpt2_jax/CABLE_PAPER_NOTES.md). grad_accum_steps = total_batch_size // (B * seq_len *
n_devices) -- on this port's 4-chip TPU v4-8 (not 8 GPUs), the SAME formula with the SAME B and
total_batch_size naturally yields a different grad_accum than the paper's own run (more chips
would mean fewer accum steps, fewer chips means more): B=16, total_batch_size=524288, seq_len=1024,
n_devices=4 -> grad_accum_steps = 524288 // (16*1024*4) = 8 (paper's own 8-GPU run: grad_accum=4
for this same B/total_batch_size). This IS the faithful match -- same B, same total_batch_size,
same formula, device-count is the only thing that differs from the paper's own hardware.

use_flash_attention=True (this port's own addition, not in Cable's original) is required to fit
batch_size=16 in a v4-8 chip's 30.75GB HBM -- see docs/status_tpu.md's OOM history.

**batch_size=16 OOM'd at this total_batch_size (grad_accum=8)** even with flash-attention+bf16,
~340MB short (24.47G needed / 24.13G free) -- confirmed the fused-grad-accum jax.lax.scan wrapper
(train_gpt.py's grad_accum_step) needs more headroom than the grad_accum=1 case ever did (that
path never invokes the scan). batch_size=16 DOES fit at grad_accum=1 (i.e. total_batch_size=65536,
not this file's 524288) -- see medium_bf16_b16ga1 in docs/status_tpu.md. At this file's
total_batch_size, use --batch-size 8 (grad_accum=16) instead, which fits comfortably.

steps_per_epoch = 10_000_000_000 // 524288 = 19073.

    uv run python gpt2_jax/train_gpt.py --config configs/gpt2_jax/medium_rope_default.py
"""
model = "medium"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
use_flash_attention = True
batch_size = 16
total_batch_size = 524288

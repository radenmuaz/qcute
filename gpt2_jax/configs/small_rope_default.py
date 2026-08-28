"""gpt2_jax.train_gpt config: Cable's own exact defaults, "small" model (n_layer=12, n_head=12,
n_embd=768, ~124M params), rope pos_method, 1 epoch over FineWeb-Edu-10B at Cable's own
total_batch_size=524288 (2**19 tokens), lr_peak=6e-4, batch_size=32/device (Cable's own
MICRO_BATCH_SIZES["small"]).

Same grad_accum-formula note as medium_rope_default.py: paper's 8x H100 run gets
grad_accum=524288//(32*1024*8)=2 for this B/total_batch_size; this port's 4-chip TPU v4-8 gets
grad_accum=524288//(32*1024*4)=4 -- same B, same total_batch_size, same formula, device count is
the only difference from the paper's own hardware.

use_flash_attention=True (this port's own addition) -- small is much smaller than medium, likely
doesn't strictly need it to fit memory, but kept on for consistency with the medium config.

steps_per_epoch = 10_000_000_000 // 524288 = 19073.

    uv run python gpt2_jax/train_gpt.py --config configs/gpt2_jax/small_rope_default.py
"""
model = "small"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
use_flash_attention = True
batch_size = 32
total_batch_size = 524288

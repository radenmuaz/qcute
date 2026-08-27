"""gpt2_jax.train_gpt config: Cable's own exact defaults, "medium" model (n_layer=24, n_head=16,
n_embd=1024, ~354M params -- Cable's README notes ~1.5GB checkpoint for this shape), rope
pos_method, 1 epoch over FineWeb-Edu-10B at Cable's own total_batch_size=524288 (2**19 tokens),
lr_peak=6e-4, batch_size=16/device (Cable's own MICRO_BATCH_SIZES["medium"]).

**Not yet smoke-tested against this port's memory ceiling** (unlike tiny's batch_size=64, which
was empirically verified: 96 needs 37.0G/chip, 128 needs 49.3G, both exceed the 30.75G available
on a v4-8 chip) -- run a short --max-steps smoke test before trusting the full run.

steps_per_epoch = 10_000_000_000 // 524288 = 19073.

    uv run python gpt2_jax/train_gpt.py --config configs/gpt2_jax/medium_rope_default.py
"""
model = "medium"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"

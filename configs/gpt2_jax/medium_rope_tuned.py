"""gpt2_jax.train_gpt config: same "medium" model as medium_rope_default.py, but a "tuned"
batch/lr pair: batch_size stays at Cable's own default (16/device, no OOM ceiling probe done yet
for this shape), and the tuning lever is total_batch_size (doubled, via grad_accum_steps -- costs
time not memory) plus a sqrt-scaled peak_lr to match (6e-4 * sqrt(2) =~ 8.5e-4).

Same caveat as the other _tuned.py configs: this does NOT make the run faster (throughput is set
by batch_size=16's per-step cost, unchanged here) -- it's a training-dynamics comparison against
medium_rope_default.py in the same wall-clock budget, not a speed tuning.

steps_per_epoch = 10_000_000_000 // 1048576 =~ 9536.

    uv run python gpt2_jax/train_gpt.py --config configs/gpt2_jax/medium_rope_tuned.py
"""
model = "medium"
pos_method = "rope"
dataset_dir = "data/fineweb-edu-10B"
batch_size = 16
total_batch_size = 2 * 524288  # 1,048,576
lr = 6e-4 * (2 ** 0.5)  # ~8.485e-4

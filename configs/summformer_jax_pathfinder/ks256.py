"""summformer_jax_pathfinder/train_pathfinder.py config: Pathfinder resolution 256 (each example
is 256*256+1=65537 tokens -- 65536 pixels + 1 label). Same "gpt2-tiny-like but deep enough"
d_model=512/n_heads=8/n_layers=2 as ks32.py, but Ks=(64,32,32) (n_fuse=3) instead: cum(Ks)=65536
still reaches the full image length, using fewer/bigger pooling steps so each intermediate code
sequence stays short enough for its OWN (unwindowed) self-attention to be cheap -- level-0's
output (length 65537) pools straight down to 1024 in one stage, then 32, then 1; only the level-0
pass itself (and its 3 refinement reruns) ever touches the full 65537 length.

attn_window=2048 is NOT optional here: dense self-attention at L=65537 would be ~4.3 billion
score-cells per head per layer, infeasible at any d_model. Bounding level-0's own self-attention to
a 2048-token local window keeps that cost to O(L*window) instead of O(L^2); long-range binding still
happens through the hierarchy (each level-1 code already has up to 2048 tokens of local context
baked in from level-0's windowed pass, then level-1's own -- still dense, only length 1024 -- self-
attention lets those locally-informed summaries interact globally in one shot). fuse_window is left
unbounded: cross-attention cost is already small here since KV shrinks fast (1024/32/1 keys), so
there's no memory pressure on that axis to bound further.

batch_size=1 (T=65537 is already huge per example), n_devices=4 -> B*T*n_devices=262148;
total_batch_size set to that exact value (grad_accum=1). 160,000 train examples * 65537 tokens =
~10.5B tokens/epoch -> ~40,000 steps/epoch at this total_batch_size, likely far more than practical
for a first pass -- max_steps below caps a starting run at 2000 steps; raise it once throughput is
known (see docs/status_tpu.md's own "watch actual elapsed time early" guidance).

    uv run python summformer_jax_pathfinder/train_pathfinder.py --config configs/summformer_jax_pathfinder/ks256.py
"""
Ks = "64,32,32"
d_model = 512
n_heads = 8
n_layers = 2
pos_method = "rope"
vocab_size = 258
sequence_length = 65537
attn_window = 2048
dataset_dir = "data/pathfinder256"
batch_size = 1
total_batch_size = 262148
max_steps = 2000
eval_every = 50
eval_steps = 4

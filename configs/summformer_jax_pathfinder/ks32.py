"""summformer_jax_pathfinder/train_pathfinder.py config: Pathfinder resolution 32 (each example is
32*32+1=1025 tokens -- 1024 pixels + 1 connected/disconnected label, see
scripts/generate_pathfinder.py). "gpt2-tiny-like but deep enough": d_model=512/n_heads=8 matches
CABLE_PAPER_NOTES.md's gpt2-tiny, n_layers=2 keeps each level's own cost small, but
Ks=(4,4,4,4,4) (n_fuse=5) makes cum(Ks)=1024 -- reaching the full image length by the last fuse
stage, so every position has a path to global context through the cascade (the actual mechanism
Pathfinder is designed to require). effective_depth = n_layers*(1+2*n_fuse) = 2*11 = 22, deeper
than gpt2-tiny's own 6 layers despite the narrow width -- deliberately, since Pathfinder needs
long-range binding a shallow model can't do regardless of receptive-field tricks.

L=1025 is small enough that dense (unwindowed) attention is cheap everywhere -- attn_window left
unbounded here (contrast with ks256.py, where it's mandatory).

batch_size=16, n_devices=4 (v4-8) -> B*T*n_devices=16*1025*4=65600; total_batch_size set to that
exact value (grad_accum=1) for a straightforward first-pass config. 160,000 train examples *
1025 tokens = 164,000,000 tokens/epoch -> ~2500 steps/epoch at this total_batch_size.

    uv run python summformer_jax_pathfinder/train_pathfinder.py --config configs/summformer_jax_pathfinder/ks32.py
"""
Ks = "4,4,4,4,4"
d_model = 512
n_heads = 8
n_layers = 2
pos_method = "rope"
vocab_size = 258
sequence_length = 1025
dataset_dir = "data/pathfinder32"
batch_size = 16
total_batch_size = 65600
num_epochs = 1
eval_every = 200
eval_steps = 10

"""3-level (trunk + 2 fuse-stages), mixed-dim: trunk is cheap/shallow (runs O(T) times), code-LMs
are wider/deeper (run O(T/K) times, amortized cheap despite the extra width/depth). Smart-guess
sizing, not yet tuned against a real bpb target -- goal is prefill+generation fast enough for
~15s/image on TPU once the KV-cache (still pending) lands, not accuracy yet.

Trunk: d_model=128, n_heads=4, n_layers=3, main_window=8 (small per "4 or 8, smaller better").
Fuse-stage 1 (finer, more frequent): K=8 -> code length 12288/8=1536, code_d_model=256 (2x trunk),
  code_n_heads=4, code_n_layers=2, inserted after layer 1 (early).
Fuse-stage 2 (coarser, less frequent): K=64 (8*8) -> code length 12288/64=192, code_d_model=512
  (4x trunk), code_n_heads=8, code_n_layers=4 (wide+deep, amortized over 64 trunk positions),
  inserted after layer 3 (i.e. after all trunk layers).
Reference point: gpt2-tiny is n_layer=6, n_head=8, n_embd=512 (gpt2_jax/train_gpt.py's
MODEL_SHAPES) -- this trunk is much smaller (cheap by design), the coarse fuse-stage 2 code-LM
matches gpt2-tiny's width exactly.

mtp_heads=24 (3*8): scaled-up multi-token-prediction per the "modern MTP + speculative decoding"
plan (docs/image_gen_design.md's "where this left off" section) -- 23 extra heads (main head
predicts t+1, extra_heads[i] predicts t+(i+2)) trained as an auxiliary loss now; the eventual
point is using these as draft heads for speculative decoding, not yet wired up.

uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/image64_mixdim.py --resolution 64 --steps 20
"""
pos_method = "rope"
vocab_size = 256
d_model = 128
n_heads = 4
n_layers = 3
main_window = 24
context_len = 12288
fuse_stages = (
    ((-1, 1), (8, -1), (2, 256, 4)),
    ((-1, 3), (64, -1), (4, 512, 8)),
)
mtp_heads = 24
batch_size = 4

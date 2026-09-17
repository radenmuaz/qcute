"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/overfit1.py

TODO/KIV -- streaming "lag" (chunked one-shot decode, e.g. wait 512px then decode in one shot,
repeat), chat 2026-09-17:
1. KIV: `decode_chunk_blocks` config -- decouple chunk/lag granularity from decoder_ncodes.
   Groups within the same chunk see the whole chunk (bidirectional, fullctx-style but scoped to
   the chunk not the whole seq); earlier chunks causal/fixed, later invisible. chunk=G recovers
   current streaming=True, chunk=n_blocks recovers fullctx/streaming=False. Real training-time
   change (new EncDecLevel field + rework Wg/ctx_windows/rope_ctx_g in
   decode_logits_and_target_pardec), not just a knob.
2. decoder_ncodes IS ALREADY the lag/chunk-size knob (zero new code): Kspan=G*K per one-shot
   call. For 32x32 (1024px), strides=(4,4) (K0=K1=4), sizing a call to exactly one 16x16 patch
   (256px = 64 level0-codes = 16 level1-codes): decoder_ncodes=(64, 16). Zorder makes a
   contiguous 256-run = one 16x16 quadrant, so both levels' one-shot call aligns to the same
   patch; 1024/256=4 patches (2x2 grid), evenly divides both levels' code counts.
   - vs baseline decoder_ncodes=(1,1): level0 chunk=4px, level1 chunk=16px-equiv, 256/64 calls.
   - vs decoder_ncodes=(2,2): level0 chunk=8px, level1 chunk=32px-equiv, 128/32 calls.
   - (64,16) = 4 calls per level to cover the whole image, each = one real 16x16 patch.
   Caveats: needs token_head_type="linears"/"diffusion" (not "ar") for the call to actually be
   one-shot at inference, not sequential-within-group via KV-cache; decoder_ncodes this
   different from training is a real train/inference mismatch (RoPE/padding/attention patterns
   tuned to a specific group size) -- validate or retrain at the target chunk size.
   No extra "waiting" beyond normal causal generation: Wg (ncodes_window/streaming, unchanged)
   still governs how far back a call can see; decoder_ncodes only changes how much gets produced
   per call.
3. Fastest to test (no code change): keep training as-is, buffer arrivals at inference and call
   decode_generate_pardec with a bigger decoder_ncodes only when serving -- it's a plain Python
   int at call time, not baked into weights. Same train/inference mismatch caveat as #2, but pay
   it only at serving time; cheapest way to get a first signal before committing to a retrain.
"""

# --- model ---
img_size = 32
# d_model = (128, 128)
d_model = (256, 256)
n_layers = (2, 2)
n_heads = (2, 2)
n_kv_heads = (None, None)
# strides = (4, 4)
code_vocab = (256, 256)
pq_chunks = (3, 3)
mlp_mult = 4
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.0
mse_weight = 0.0
entropy_weight = 0.0

strides = (4, 4)
decoder_ncodes = 1
ncodes_window = 0
# ncodes_window = -1
attn_lookahead = 0
decode_past = 16
decode_future = 16
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 1.0
# gumbel_temperature = 0.1
gumbel_at_inference = False

mse_softmax_tau = 1.0
level_drop = 0.5
quantize_drop = 0.5
# feedback_p = 0.5
feedback_p = 0.0


init_scheme = "llama"
use_xsa = True
use_attn_sink = True
precision = "fp32"
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
pq_dim = (128, 128)
token_dim = (128, 128)
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"
eval_gen_train = True

# --- training ---
batch_size = 16
val_batch_size = 8
phase_steps = (int(1e4), int(1e5))
seed = 0
# warmup_steps = 2
# train_subset_n = None
train_subset_n = 1000
gen_eval_every_step = 1000

grad_clip = 10.0
lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_step = int(1e4)
warmup_steps = 100
# lr_min_epoch = 50
# lr_min_epoch = 400
weight_decay = 1e-2
optimizer = "adamw"
optimizer_kwargs = {}

# wa_mode = "none"

wa_mode = "ema"
wa_every_step = 100
wa_ema_decay = 0.9
wa_verbose = False

# wa_mode = "wma"
# wa_every_epoch = 1
# wa_stack_size = 5
# wa_wma_weights = (5.0, 4.0, 3.0, 2.0, 1.0)

# --- logging ---
log_every = 100
ckpt_every_step = 1000
ckpt_keep = 1

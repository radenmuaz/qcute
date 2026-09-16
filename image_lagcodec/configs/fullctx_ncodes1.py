"""
uv run python3 -m image_lagcodec.run_lagcodec --config image_lagcodec/configs/fullctx_ncodes1.py

TRUE FULLCTX preset (chat 2026-09-15, renamed from fullctx_parallel.py, fixed twice before that):
no streaming -- the entire encoder code sequence is already known upfront, nothing more is
coming. decoder_ncodes=1 (finest granularity, "one seed one token" -- one ctx code per group) +
streaming=False + ncodes_window=-1 (every single-code group sees the SAME full real ctx directly,
no padding, no causal-growing-prefix restriction) gives maximum parallelism with full context,
cheaply. Was ncodes_window=-1 with default streaming=True -- WRONG for this config: causal
streaming still restricts each group to a growing prefix, so group 0 got padded to (n_groups-1)*G
fake blocks for just 1 real one, OOM'ing at batch_size=16 (48G needed vs 30.75G available). Was
then ncodes_window=-2 (a now-removed sentinel, replaced by the streaming=False + ncodes_window=-1
combo -- same runtime behavior, just a cleaner two-axis config surface: window extent x
streaming). Contrast with stream16_unbounded.py/stream16_window2.py (streaming=True, fixed
16-code chunks, genuinely needs the causal growing prefix) and the OTHER fullctx extreme
(fullctx_ncodesfull.py: decoder_ncodes=n_blocks, single group, falls back to the fast original
slow sequential AR automatically).
"""


# --- model ---
img_size = 32
d_model = (256, 256, 256, 256, )
# n_layers = (2, 2, 2, 2,)
n_layers = (4, 4, 4, 4,)
n_heads = (2, 2, 2, 2,)
n_kv_heads = (None, None, None, None,)
strides = (4, 4, 4, -1)
code_vocab = (256, 256, 256, 256,)
pq_chunks = (3, 3, 3, 3,)
mlp_mult = 2
rope_base = 10000.0

ntp_weight = 1.0
mtp_weight = 0.1
mse_weight = 1.0
entropy_weight = 0.01

decoder_ncodes = 1
ncodes_window = -1   # "all" -- combined with streaming=False below, true fullctx: every
# single-code group sees the SAME full real ctx directly, no padding
streaming = False
weight_sharing = False
# weight_sharing = True
curriculum_mode = "no_freeze"
quantize_mode = "gumbel"
gumbel_temperature = 0.1
gumbel_at_inference = False
cascade_rollout_drop = 0.5
quantize_drop = 0.5
init_scheme = "llama"
use_xsa = True
pq_dim = (128, 128, 128, 128)
# pq_dim = (64, 64, 64, 64,)

byte_group = 3
token_head_type = "ar"
token_dim = (128, 128, 128, 128)
# token_dim = (64, 64, 64, 64,)
token_n_heads = 2
mtp_horizon = 1
# mtp_mode = "ar"
mtp_mode = "parallel"
traversal = "zorder"

# --- training ---
batch_size = 4   # chat 2026-09-15: -2 (fullctx) fixes CTX SEMANTICS vs -1 (no more causal-prefix
# restriction) but does NOT reduce memory -- both end up at the same Wg=n_blocks_p width for the
# naive uniform-shape-across-groups implementation (confirmed: -2 OOM'd too, 49.19G). The real
# OOM driver is decoder_ncodes=1 -> n_groups=256 -> B2=batch_size*n_devices*n_groups=16384 dense
# attention rows -- inherent to this naive/dense approach at small G, unrelated to -1 vs -2.
val_batch_size = 16
phase_epochs = (100, 100, 100, )
warmup_steps = 1000
grad_clip = 10.0
seed = 0
# warmup_steps = 2
train_subset_n = None

lr = 1e-3
lr_schedule = "cosine"
lr_min = 1e-5
lr_min_epoch = 80
# lr_min_epoch = 400
weight_decay = 1e-4
optimizer = "adamw"
optimizer_kwargs = {}

wa_mode = "none"

# --- logging ---
log_every = 100
gen_eval_every_epoch = 10
ckpt_every_epoch = 10
ckpt_keep = 1
qual_gen_n = 16

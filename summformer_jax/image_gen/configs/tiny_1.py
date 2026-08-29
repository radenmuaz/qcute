"""3-level (trunk + 2 fuse-stages), mixed-dim: trunk is cheap/shallow (runs O(T) times), code-LMs
are wider/deeper (run O(T/K) times, amortized cheap despite the extra width/depth).

epochs -> steps: one step processes `batch_size` (per-device) * `n_devices` images (train.py's
jax.pmap data-parallel loop, one micro-batch per device per step -- see train.py's own docstring).
    steps_per_epoch = ceil(NUM_TRAIN_IMAGES / (batch_size * n_devices))
    steps           = epochs * steps_per_epoch
`n_devices` is a config-write-time guess (v4-8 node = 4 addressable chips, confirmed via
jax.local_device_count() on tpu8) -- train.py itself reads the REAL device count at runtime
(jax.local_devices()) and uses all of them regardless of this guess; only `steps`/`save_every`
below (computed ahead of time so the config is a plain, self-contained number) assume it.

uv run python summformer_jax/image_gen/train.py --config summformer_jax/image_gen/configs/tiny_1.py
"""
import math

NUM_TRAIN_IMAGES = 1_281_167  # ILSVRC/imagenet-1k train split, confirmed via prep_imagenet64 full run, 2026-08-29
N_DEVICES = 4                 # v4-8 node guess -- see docstring, train.py uses the real count regardless
BATCH_SIZE = 16                # per-device

STEPS_PER_EPOCH = math.ceil(NUM_TRAIN_IMAGES / (BATCH_SIZE * N_DEVICES))
EPOCHS = 100
EVAL_EVERY_EPOCHS = 10
SAVE_EVERY_EPOCHS = 10

pos_method = "rope"
vocab_size = 256
d_model = 256
n_heads = 4
n_layers = 2
main_window = 24
context_len = 12288
fuse_stages = (
# ((src, dst), (stride, cross_attn_window), (code_n_layers, code_d_model, code_n_heads, code_window))
    ((-1, 1), (8, 24), (2, 256, 4, 24)),   # window=24 >= stride=8 -- real windowed cross-attn
    ((-1, 2), (8, 24), (2, 256, 4, 24)),   # stride=8 (was 64) -- matches stage 1 so window=24 valid
    # dst=2 (was 3) since n_layers is now 2, dst=3 would silently never fire (insertions checked
    # only for depth<=n_layers)
)
mtp_heads = 24
batch_size = BATCH_SIZE
resolution = 64

steps = EPOCHS * STEPS_PER_EPOCH
steps_per_epoch = EVAL_EVERY_EPOCHS * STEPS_PER_EPOCH   # train.py's eval cadence -- eval every 10 real epochs
save_every = SAVE_EVERY_EPOCHS * STEPS_PER_EPOCH

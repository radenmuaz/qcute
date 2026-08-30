"""ViT-Tiny-param-count-sized classifier (~4.66M params, ViT-Ti/16 reference is ~5.7M),
d_model=128/n_layers=16 (sizing-checked against a few (d_model, n_layers) combos before landing
here -- see chat). Attention windows all small (main_window=24, code_window=24). Stages 1/2 keep
cross-attention windowed at their own stride (minimum valid -- window must be >= stride or most
queries see zero code context, see summformer.py's windowed_cross_attention docstring): their
code-sequence length (T/stride = 9408, 2352) makes dense cross-attention there expensive (this is
what caused the earlier 165GB OOM when ALL stages were left unbounded, confirmed directly).
Stage 3 (stride=256, code length 588) is left dense/unbounded (-1) instead: empirically confirmed
(check_block_locality) that a *windowed* stage-3 (matching image_gen/tiny_1.py's OOM-avoidance
convention of window=stride) leaves the classifier's effective receptive field cut off somewhere
in [2000, 10000) out of the full 150528 -- i.e. blind to 93-99% of the image -- and that widening
window alone doesn't fix it without dense-scale memory anyway, so at stage 3's cheap code length
going fully dense there is strictly better: same coverage as a huge window, far less memory (S=588
is small enough that O(T*S) is cheap even dense).

Optimizer recipe approximates DeiT (Touvron et al. 2021) -- the standard from-scratch-on-
ImageNet1k recipe for this exact param scale (ViT-Ti/16): AdamW, weight_decay=0.05, base_lr=5e-4
scaled by batch/512, 5-epoch linear warmup + cosine decay, ~300 epochs. NOTE: DeiT's own reported
DeiT-Tiny top-1 is ~72.2%, not 75%, and that's WITH heavy augmentation (RandAugment/Mixup/CutMix/
stochastic depth/label smoothing) this pipeline doesn't yet implement (only random-resized-crop +
flip) -- see train.py's own docstring/chat discussion. 75% top-1 at this exact param count without
that augmentation stack is optimistic; recalibrate expectations or add augmentation later.

    uv run python summformer_jax/image_classification/train.py --config summformer_jax/image_classification/configs/tiny_vit_like.py --shard-dir /dev/shm/imagenet_raw
"""
IMAGE_SIZE = 224
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 16

pos_method = "rope"
vocab_size = 256
d_model = D_MODEL
n_heads = N_HEADS
n_layers = N_LAYERS
main_window = 24
context_len = IMAGE_SIZE * IMAGE_SIZE * 3
num_classes = 1000
bidirectional = False
mtp_heads = 1  # no MTP extra heads -- pure classification, no next-token prediction

fuse_stages = (
    ((-1, 4), (16, 16), (1, D_MODEL, N_HEADS, 24)),
    ((-1, 10), (64, 64), (1, D_MODEL, N_HEADS, 24)),
    ((-1, N_LAYERS), (256, -1), (1, D_MODEL, N_HEADS, 24)),
)

batch_size = 32  # per-device
base_lr = 5e-4  # DeiT's own value @ batch=512, scaled by batch/512 in train.py
warmup_epochs = 5.0
weight_decay = 0.05
num_epochs = 100.0  # scaled down from DeiT's 300 -- recalibrate once real it/s is known

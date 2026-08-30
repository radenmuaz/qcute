"""Bidirectional variant of tiny_vit_like.py using MultiScanClassifier (multiscan_classifier.py)
instead of the plain SummClassifier -- forward (row_major) and reverse (row_major_reverse) passes
share ONE backbone (single group containing both scan names), mean-pooled together, one linear
head. This is the "bidirectional, share weight forward backward" case requested after
tiny_vit_like.py's single-scan run confirmed stable (see chat/docs/status_tpu.md).

Same backbone sizing/fuse_stages/receptive-field fix as tiny_vit_like.py (stage 3 dense/unbounded
cross-attention window, see that file's own docstring for the full rationale) -- only the
classification head changes (MultiScanClassifier's mean-pool-then-linear over 2 scan passes,
instead of SummClassifier's single last-token pass).

    uv run python summformer_jax/image_classification/train.py --config summformer_jax/image_classification/configs/tiny_vit_like_bidir_shared.py --shard-dir /dev/shm/imagenet_raw
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
mtp_heads = 1  # no MTP extra heads -- pure classification, no next-token prediction

fuse_stages = (
    ((-1, 4), (16, 16), (1, D_MODEL, N_HEADS, 24)),
    ((-1, 10), (64, 64), (1, D_MODEL, N_HEADS, 24)),
    ((-1, N_LAYERS), (512, -1), (1, D_MODEL, N_HEADS, 24)),
)
# stride bumped 256->512 vs tiny_vit_like.py's single-scan version: bidirectional runs the
# backbone twice per step (forward+reverse), roughly doubling peak HBM, and batch_size=1 there
# still exceeded 30.75G by a hair (18.5K over) -- halving stage 3's dense code-sequence length
# (588->294) trims that back while keeping the stage fully dense (still full receptive-field
# coverage, just coarser code-token granularity).

image_size = IMAGE_SIZE
channels = 3
scan_groups = (("row_major", "row_major_reverse"),)  # 1 group, 2 scans -> 1 shared backbone
output_mode = "mean_pool"

batch_size = 32  # per-device
base_lr = 5e-4  # DeiT's own value @ batch=512, scaled by batch/512 in train.py
warmup_epochs = 5.0
weight_decay = 0.05
num_epochs = 100.0  # scaled down from DeiT's 300 -- recalibrate once real it/s is known

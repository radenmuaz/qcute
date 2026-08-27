"""v1_stack_simplex/ks21_v64_pq4_overfit10k: same as ks21_v256_pq1.py but vocab=64 (per-chunk width),
pq_chunks=4 -- 4 independent 64-way softmaxes (24 bits combinatorial, 8 bits nominal
head-cost-per-chunk) instead of one 256-way softmax (8 bits). Paired PQ-vs-no-PQ comparison,
same total code width (256), directly following the v1 autoencoder-decode reframing where
codebook capacity is now a hard per-block reconstruction constraint (2 bytes = 16 bits needed
for Ks=(2,1), see docs/qcute_lagcodec_plan.md's open-risks note), not just a soft compression knob.

uv run python -m qcute.qcute_lagcodec.qcute_lagcodec --decoder_type stack --config configs/lagcodec/lagcodec_stack_simplex/ks21_v64_pq4_overfit10k.py

# plot after training:
uv run python scripts/plot_run.py logs/v1_stack_simplex_ks21_v64_pq4_overfit10k
"""
from pathlib import Path

run_name = "v1_stack_simplex_ks21_v64_pq4_overfit10k"
decoder_type = "stack"  # StackDecoder -- the current default lineage (chat 2026-08-20:
# StackDecoderV1 is now legacy, memory-expensive relative to this one, see
# encode_like_self_attn_decode/seed_query_decode's docstrings)
Ks = (2, 1)
d_model = 256
n_layers = 1
context_len = 256
attn_window = -1
code_hard = True
code_sample = False
quant_type = "simplex"
vocab = 64
pq_chunks = 4
input_preset = 8
output_preset = 8
entropy_reg_weight = 0.0

data = Path("datasets/enwik8_1M.gz")
n_bytes = 10000
val_frac = 0.1

steps = 1000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 100
cosine_decay = False
log_every = 20
eval_every = 50
eval_batches = 5

qual_gen_bytes = 0  # check_gen_consistency/check_roundtrip_consistency/check_decode_modes (called
# whenever qual_gen_bytes>0) are still StackDecoderV1-specific (reference bos_interleaved_self_attn/
# own_block_cross_attn_decode and the bb_self/bb_cross stage_lms split directly, on the Decoder base
# class -- would crash against StackDecoder's different stage_lms structure). StackDecoder's own
# generation fix (StackDecoder._generate_blockwise, chat 2026-08-20) is validated separately via
# check_blockwise_gen_consistency, called manually/from a script, not wired into this training loop.

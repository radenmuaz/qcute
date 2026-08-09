"""v4.4 packed-sequence decode: n_levels=1, Ks=(1,), decode_pack_mode="interleave", chunked
attention (LevelLM._packed_decode_forward_chunked). Degenerate self-conditioning case (no level
above level 0): level 0's own code c_0 is fed back as the "code above" via RefineLM._run's
n_levels==1 fallback (source_c = c_list[0], decode_K = Ks[0] = 1).

context_len=512 (not the production 1024) -- windowed attention only requires context_len to be
a multiple of attn_window (32), and reducing it here keeps a first v4.4 training run cheap while
its actual step-time profile at scale is still being characterized (see
scripts/test_v4_4_chunked_decode.py, which verified chunked==dense exactly and that chunked's
linear-in-L scaling wins decisively over dense's quadratic-in-L cost by context_len=1024, where
dense was skipped in that benchmark as impractical).

    uv run python -m qcute.qcute_refine_v4_4 --config configs/qcute_refine_v4_4_l1_k1.py

    # watch live:
    tail -f logs/qcute_refine_v4_4_l1_k1/run.log
"""
from pathlib import Path

Ks = (1,)
d_model = 256
n_layers = 2
context_len = 512
attn_window = (32,)
decode_pack_mode = "interleave"
decode_chunked = True

data = Path("datasets/enwik8_1M.gz")
val_frac = 0.1

steps = 4000
batch_size = 16
lr_peak = 6e-4
warmup_steps = 500
cosine_decay = False
log_every = 50
eval_every = 100
eval_batches = 20

qual_gen_bytes = 64
qual_prompt_bytes = 64

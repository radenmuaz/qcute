"""CPU, tiny-scale forward and forward+backward wallclock comparison of
qcute_refine_v2's four interchangeable "predict dq bits from a hidden
vector" implementations:

  - linear:    a single plain nn.Linear(d_model, dq) — the "independent"
               baseline (code_head_mode="independent"/tok_head_mode=
               "linear"), NO chain-rule cross-bit conditioning at all.
  - attn:      BitPredictHeadAttn (bit_head_class="attn", default) —
               causal self-attention over the dq-bit chain.
  - conv:      BitPredictHeadConv (bit_head_class="conv") — causal
               window read over the bit-embedding sequence, "matmul"
               (default reparam) or "conv1d" (original nn.Conv1d) impl.
  - ssm:       BitPredictHeadSSM (bit_head_class="ssm") — linear-decay
               recurrence, computed as one batched matmul when
               teacher-forced.

Also sweeps Config.bit_inner_downsample (1x/2x/4x) for attn/conv/ssm —
projects the incoming hidden vector down to d_model//downsample once,
then runs every internal chain op at that smaller width instead of the
full d_model (1x is a no-op, identical to pre-flag behavior).

Forced to CPU explicitly (never contend with a live MPS training job).
Tiny d_model/dq/batch by design — this isolates per-call op overhead,
not a realistic training-step budget; see docs/status.md's own
~200x-1800x chain-vs-linear finding (that number was from an earlier
ad hoc check, this script is the reproducible version of it).

    uv run python scripts/bench_bit_heads.py
"""
import sys, time
sys.path.insert(0, '/Users/muaz/code/qcute')
import torch
import torch.nn as nn

from qcute.qcute_refine_v2 import BitPredictHeadAttn, BitPredictHeadConv, BitPredictHeadSSM

device = 'cpu'
torch.manual_seed(0)

N = 32          # batch of hidden vectors (flattened B*T, as these heads are called in training)
D_MODEL = 32    # tiny d_model
N_TIME = 50
N_WARMUP = 5


class LinearHead(nn.Module):
    """The "independent" baseline: no chain-rule conditioning, one shot."""
    def __init__(self, d_model: int, dq: int):
        super().__init__()
        self.fc = nn.Linear(d_model, dq)

    def forward(self, h: torch.Tensor, true_bits: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(h)


def bench(name: str, model: nn.Module, h: torch.Tensor, true_bits: torch.Tensor):
    n_params = sum(p.numel() for p in model.parameters())

    # forward-only (eval, no_grad) — inference-shaped cost, PARALLEL/teacher-forced
    # (true_bits passed -> _forward_fixed: one batched call, no python-level
    # sequential dependency between bit positions)
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(h, true_bits)
        t0 = time.perf_counter()
        for _ in range(N_TIME):
            model(h, true_bits)
        fwd_dt = (time.perf_counter() - t0) / N_TIME

    # forward+backward (train) — training-step-shaped cost, same PARALLEL path
    model.train()
    for _ in range(N_WARMUP):
        model.zero_grad(set_to_none=True)
        out = model(h, true_bits)
        out.sum().backward()
    t0 = time.perf_counter()
    for _ in range(N_TIME):
        model.zero_grad(set_to_none=True)
        out = model(h, true_bits)
        out.sum().backward()
    fwdbwd_dt = (time.perf_counter() - t0) / N_TIME

    # DECODE: true_bits=None -> _forward_loop, greedy, one python loop
    # iteration per bit position, each iteration feeding back its own
    # just-decided bit — the actual autoregressive generation-time cost,
    # inherently sequential for the chain heads. LinearHead has no chain
    # dependency at all (all dq bits predicted in one shot regardless of
    # true_bits), so its "decode" number is identical to its forward one
    # by construction, not a separate code path.
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            model(h, None)
        t0 = time.perf_counter()
        for _ in range(N_TIME):
            model(h, None)
        decode_dt = (time.perf_counter() - t0) / N_TIME

    print(f"{name:10s} params={n_params:6d}  fwd={fwd_dt*1e3:9.4f}ms  fwd+bwd={fwdbwd_dt*1e3:9.4f}ms  "
          f"decode={decode_dt*1e3:9.4f}ms  decode/fwd={decode_dt/fwd_dt:7.2f}x")
    return fwd_dt, fwdbwd_dt, decode_dt


for dq in (8, 16, 32):
    print(f"\n=== dq={dq}  (d_model={D_MODEL}, N={N}, CPU) ===")
    h = torch.randn(N, D_MODEL)
    true_bits = (torch.rand(N, dq) > 0.5).float() * 2 - 1  # {-1,+1}-ish

    linear_fwd, linear_fwdbwd, linear_decode = bench("linear", LinearHead(D_MODEL, dq), h, true_bits)
    convc_fwd, convc_fwdbwd, convc_decode = bench("conv(conv1d,1x)", BitPredictHeadConv(D_MODEL, dq, conv_impl="conv1d"), h, true_bits)

    print("  -- bit_inner_downsample sweep (1x/2x/4x) --")
    for ds in (1, 2, 4):
        attn_fwd, attn_fwdbwd, attn_decode = bench(f"attn {ds}x", BitPredictHeadAttn(D_MODEL, dq, downsample=ds), h, true_bits)
        conv_fwd, conv_fwdbwd, conv_decode = bench(f"conv {ds}x", BitPredictHeadConv(D_MODEL, dq, conv_impl="matmul", downsample=ds), h, true_bits)
        ssm_fwd, ssm_fwdbwd, ssm_decode = bench(f"ssm {ds}x", BitPredictHeadSSM(D_MODEL, dq, downsample=ds), h, true_bits)
        print(f"    {ds}x slowdown vs linear, train  (fwd+bwd): attn={attn_fwdbwd/linear_fwdbwd:6.1f}x  "
              f"conv={conv_fwdbwd/linear_fwdbwd:6.1f}x  ssm={ssm_fwdbwd/linear_fwdbwd:6.1f}x")
        print(f"    {ds}x slowdown vs linear, decode (loop)   : attn={attn_decode/linear_decode:6.1f}x  "
              f"conv={conv_decode/linear_decode:6.1f}x  ssm={ssm_decode/linear_decode:6.1f}x")

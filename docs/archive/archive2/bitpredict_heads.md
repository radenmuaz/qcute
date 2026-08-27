# BitPredictHead speed: linear vs. attn/conv/ssm, matmul reparam, and inner-downsample

Cross-referenced from [docs/status.md](status.md). Full technical detail
lives in `qcute/qcute_refine_v2.py`'s own class docstrings
(`BitPredictHeadAttn`/`Conv`/`SSM`, `Config.bit_conv_impl`,
`Config.bit_inner_downsample`) — this doc is the narrative/findings
summary, not a duplicate of the code-level rationale.

`scripts/bench_bit_heads.py`: CPU, tiny-scale (`d_model=32`, `N=32`)
forward and forward+backward wallclock for the "independent" `nn.Linear`
baseline (no chain-rule conditioning) vs. the three chain heads
(`bit_head_class` = `attn`/`conv`/`ssm`), across `dq` in `(8,16,32)`,
plus a `decode` mode (`true_bits=None` → `_forward_loop`, the actual
sequential-loop autoregressive-generation-time cost, distinct from the
batched teacher-forced `_forward_fixed` path used for training).

## Findings

**Train (fwd+bwd) slowdown vs. linear**: all three chain heads cost real
overhead (expected — they pay for cross-bit conditioning the linear head
skips entirely), roughly attn 11-19x, conv 9-51x, ssm 12-32x depending on
dq. **Decode (sequential loop) slowdown is far worse and roughly
dq-scaling**: attn 165-730x, ssm 54-136x (`ssm` consistently cheapest —
no dq-dependent op, just a per-step linear-decay update), and `conv`
worst-case **~3900x at dq=16 specifically** with the original `nn.Conv1d`
implementation — a backend-quirk spike (dq=8 and dq=32 are far cheaper,
~25-72x), not a real algorithmic cost.

**Fix**: `BitPredictHeadConv` gained `conv_impl` (`"matmul"`, new
default, vs. `"conv1d"`, original) — mathematically the SAME operation
(fixed causal window, weights shared across positions), just
reparametrized as `nn.Linear(kernel_size*D, D)` over a flattened window
instead of calling `nn.Conv1d` directly. Verified numerically
fixed/loop-consistent (~1e-7 diff) at every kernel size tried. Fixes the
dq=16 anomaly entirely and is faster across the board: dq=16 train
428x→14x slower than linear, decode 3876x→104x. Wired through
`Config.bit_conv_impl` + CLI; kept BOTH modes as a flag, not a
replacement (repo convention).

**Also added**: `Config.bit_inner_downsample` (1/2/4, default 1 = exact
prior behavior, no extra op/params) — projects the incoming hidden vector
down to `d_model//downsample` once via a new `in_proj`, then runs every
internal chain op (embeds/attn/conv/ssm-state/head) at that smaller width
instead of full `d_model`, for all three `bit_head_class` variants
uniformly. Verified fixed/loop-consistent at every downsample factor.
**Helps train cost** (roughly halves-to-thirds the slowdown-vs-linear
going 1x→4x, e.g. attn 25x→16x, conv 26x→12x, ssm 32x→14x at dq=32; params
drop sharply too, e.g. attn dq=32: 5345→833). **Barely moves decode
cost** and is sometimes non-monotonic there (attn dq=32: 728x→583x→653x
across 1x/2x/4x) — decode's dominant cost is Python-loop/dispatch
overhead from dq sequential calls, not per-call matmul width, so
shrinking the matmuls doesn't touch the actual bottleneck. Net:
`bit_inner_downsample` is a solid free-ish training-time lever, not a fix
for generation-time cost — `ssm` remains the cheapest decode option
regardless of downsample.

## Relevance to observed run speeds

`qcute_refine_v1`'s own module uses `BitPredictHead` chain-mode NTP
heads throughout (unlike `qcute_refine_v2`'s `code_head_mode=
"independent"` runs) — consistent with it being by far the slowest
completed run (0.165 mean it/s, see the ablation-family comparison table
in [docs/status.md](status.md)).

# `torch.compile` on `qcute_refine_v2`

Cross-referenced from [docs/status.md](status.md). Covers the `--compile`
CLI flag on `qcute/qcute_refine_v2.py` (default `False`, no-op unless
passed) — root cause of why the naive approach was slower than eager, the
real fix, and the final MPS speed verdict.

## Three attempts

**Attempt 1 (wrong)**: `model = torch.compile(model)` on the whole
`RefineLM`. Measured net SLOWER than eager (200 real steps,
`configs/qcute_refine_rope.py`, MPS: eager steady ~2.99 it/s, compiled
never caught up, ~2.5 it/s and still climbing). Root cause found via
`TORCH_LOGS=recompiles`: `RefineLM.forward` took the raw training-step
int `step` and branched on its exact value inside `n_active_levels()`
(gating `layer_warmup_steps`) — dynamo guards on that exact int and
recompiled almost every single step, even for configs that never set
`layer_warmup_steps` at all, since `step` still reached the compiled
function as a live-changing value.

**Attempt 2 (workaround, since removed)**: assert `layer_warmup_steps`
empty, pass `step=None` into the model whenever compiling. Fixed the
recompile problem but disabled curriculum+compile entirely.

**Attempt 3 (the real fix, what's in the code now)**: `RefineLM.forward`
now takes `n_active: int | None` directly instead of raw `step`.
`train()`/`eval_model()` compute `n_active = model.n_active_levels(step)`
themselves, in plain eager Python, and pass that in — `step` itself never
reaches the (possibly compiled) model call. Dynamo now guards on
`n_active`, which only takes a handful of distinct values across an
entire run (one per curriculum stage transition) instead of one per
step. Verified with a real 3-stage curriculum (`layer_warmup_steps=
(5,5)`, 20 steps crossing both transitions): every recompile event tied
to a genuinely new tensor shape appearing for the first time (a new
level activating really is a different compute graph), clustered right
at the two transitions, zero guard failures mention `step` anywhere.
Correctness verified throughout (matched-seed eager vs compiled,
multiple architectures/curricula): loss diffs ~1e-6, grad diffs ~1e-9,
the normal float32 op-reordering noise floor, no regression at any
point in this process.

## Real MPS speed result

200 real steps, `qcute_refine_rope`'s non-curriculum config, so this
isn't confounded by the old recompile bug: eager ~69s/199 steps (2.88
it/s), compiled (fixed) ~80s/199 steps (2.49 it/s) — **still ~13-15%
SLOWER than eager**, even with the recompile bug genuinely gone. Likely
cause: `d_model=256` means small matmuls, and MPS/Inductor's
kernel-launch overhead + less mature fusion coverage (vs. CUDA) probably
dominates over any compile-time fusion win at this scale.

**Conclusion: `--compile` is correct and available, but not worth using
on this hardware/model-scale combination — stays off by default.**

## Related

`DecoderLevel`'s cross-attention KV mask was re-verified correct while
debugging this — see [docs/kv_contribution.md](kv_contribution.md) for
that finding (not otherwise related to `torch.compile` itself, just
discovered in the same debugging pass).

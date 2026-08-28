"""Save/load a SummTransformer checkpoint (orbax `PyTreeCheckpointer`, plain-dict pytree of
params -- matches what train_summformer.py's checkpoint step writes: `nnx.to_pure_dict(state)` on
the un-replicated (single-device) params).

Works identically on TPU or CPU -- restore_args are built from whatever `jax.devices()` reports,
so `JAX_PLATFORMS=cpu` (local inspection) and a real TPU node (resuming/eval on-device) both work
with the same call, single-device sharding either way (no cross-device resharding here -- this is
for single-chip inspection/eval, not resuming a pmap'd multi-chip run).

    uv run python -c "
    from summformer_jax.checkpoint_io import load_checkpoint
    from summformer_jax.model_summformer import Config
    m = load_checkpoint('logs/summformer_small_ablation/model_19072',
                         Config(Ks=(2,2,2), d_model=768, n_heads=12, n_layers=1))
    "
"""
from __future__ import annotations

from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx

try:  # flat import when PYTHONPATH=summformer_jax (how train_summformer.py itself is launched)
    from model_summformer import Config, SummTransformer
except ImportError:  # package-qualified import when run from the repo root instead
    from summformer_jax.model_summformer import Config, SummTransformer


def save_checkpoint(path: str | Path, state: nnx.State) -> None:
    """`state` must already be un-replicated (no leading pmap-device axis) -- callers doing
    multi-device training strip that axis first, e.g. `jax.tree.map(lambda x: x[0], params)`."""
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(Path(path).resolve(), nnx.to_pure_dict(state))


def load_checkpoint(path: str | Path, cfg: Config, *, seed: int = 0) -> SummTransformer:
    """Builds a fresh SummTransformer from `cfg` (must match the checkpoint's own architecture --
    Ks/d_model/n_heads/n_layers/vocab_size/weight_tie etc., since shapes must line up exactly) and
    restores its params in place from `path`. Returns the ready-to-use merged model."""
    model = SummTransformer(cfg, rngs=nnx.Rngs(seed))
    graphdef, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)

    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), pure)

    with ocp.PyTreeCheckpointer() as ckptr:
        restored = ckptr.restore(Path(path).resolve(), item=pure, restore_args=restore_args)

    nnx.replace_by_pure_dict(state, restored)
    return nnx.merge(graphdef, state)

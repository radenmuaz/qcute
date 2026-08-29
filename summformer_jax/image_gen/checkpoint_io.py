"""Checkpoint I/O for train.py: params + optimizer state via orbax `PyTreeCheckpointer` (pure
array pytrees, single-device -- multi-device callers strip the leading pmap-device axis first,
`jax.tree.map(lambda x: x[0], pytree)`), plus step count + dataloader position via plain `pickle`
alongside (small, irregularly-shaped data -- python ints and a numpy RNG state dict don't map
cleanly onto orbax's array-restore-args machinery, not worth fighting that for a few KB).

    <path>/params/    -- orbax PyTreeCheckpointer, nnx.to_pure_dict(state)
    <path>/opt_state/  -- orbax PyTreeCheckpointer, optax opt_state pytree
    <path>/extra.pkl   -- {"step": int, "loader_state": dict, ...}
"""
from __future__ import annotations

import pickle
from pathlib import Path

import jax
import orbax.checkpoint as ocp


def save_checkpoint(path: str | Path, params: dict, opt_state, extra: dict) -> None:
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(path / "params", params)
        ckptr.save(path / "opt_state", opt_state)
    with open(path / "extra.pkl", "wb") as f:
        pickle.dump(extra, f)


def load_checkpoint(path: str | Path, params_template: dict, opt_state_template):
    path = Path(path).resolve()
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    params_restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), params_template)
    opt_restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(sharding=sharding), opt_state_template)
    with ocp.PyTreeCheckpointer() as ckptr:
        params = ckptr.restore(path / "params", item=params_template, restore_args=params_restore_args)
        opt_state = ckptr.restore(path / "opt_state", item=opt_state_template, restore_args=opt_restore_args)
    with open(path / "extra.pkl", "rb") as f:
        extra = pickle.load(f)
    return params, opt_state, extra

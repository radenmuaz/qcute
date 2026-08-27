"""Plot train/val loss+bpb curves from a gpt2_jax/summformer_jax run's log.jsonl to a PNG.

Schema: one JSON record per line, {'step': N, 'split': 'train'|'val', 'loss': L, 'bpb': B, ...}
-- differs from scripts/plot_run.py's older qcute lineages (separate bpb/val_bpb keys, elapsed_s
not step-indexed splits). Saves <log_dir>/loss_curve.png next to the log.jsonl by default.

    uv run python scripts/plot_jax_run.py logs/<run_name>
    uv run python scripts/plot_jax_run.py logs/<run_name>/log.jsonl --out somewhere/else.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_records(jsonl_path: Path) -> list[dict]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line mid-write
    return records


def extract(records: list[dict], split: str, key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for r in records:
        if r.get("split") == split and "step" in r and key in r:
            steps.append(r["step"])
            values.append(r[key])
    return steps, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="log dir (containing log.jsonl) or the log.jsonl file itself")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    jsonl_path = args.path / "log.jsonl" if args.path.is_dir() else args.path
    out_path = args.out or jsonl_path.parent / "loss_curve.png"

    records = load_records(jsonl_path)
    if not records:
        print(f"no records in {jsonl_path}")
        return

    run_name = jsonl_path.parent.name
    fig, (ax_loss, ax_bpb) = plt.subplots(1, 2, figsize=(12, 5))

    for split, color in [("train", "tab:blue"), ("val", "tab:orange")]:
        steps, loss = extract(records, split, "loss")
        if steps:
            ax_loss.plot(steps, loss, label=split, color=color,
                         marker="o" if split == "val" else None, markersize=3, linewidth=1 if split == "train" else 1.5,
                         alpha=0.6 if split == "train" else 1.0)
        steps_b, bpb = extract(records, split, "bpb")
        if steps_b:
            ax_bpb.plot(steps_b, bpb, label=split, color=color,
                        marker="o" if split == "val" else None, markersize=3, linewidth=1 if split == "train" else 1.5,
                        alpha=0.6 if split == "train" else 1.0)

    for ax, title in [(ax_loss, "loss (nats)"), (ax_bpb, "bpb")]:
        ax.set_xlabel("step")
        ax.set_ylabel(title)
        ax.set_title(title)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        ax.grid(alpha=0.3)

    last_step = max((r["step"] for r in records if "step" in r), default=0)
    fig.suptitle(f"{run_name} (step {last_step})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path} ({len(records)} records, last step {last_step})")


if __name__ == "__main__":
    main()

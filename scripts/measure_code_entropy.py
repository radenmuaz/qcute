"""Diagnostic: measure code-usage entropy_reg and code utilization for already-trained bsq/softmax
checkpoints (quant_type="bsq" bsq_bits=16 and quant_type="softmax", both code_sample_mode="ste",
qcute_v5_concat.py and qcute_v5_stack.py) -- informs whether Config.entropy_reg_weight is worth
turning on for future runs (these checkpoints were all trained with it at its default 0.0/off).

For each checkpoint, samples several TRAIN batches then several VAL batches, runs a forward pass,
and reports per-level:
  - entropy_reg: E_batch[H(p)] - H(E_batch[p]) (see docs/bsq_entropy_reg_math.md) -- already exposed
    in RefineLM.forward's own metrics dict, no model changes needed.
  - distinct_codes_used / possible_ids: a more directly interpretable code-utilization count, via
    quant.to_ids() on the actually-quantized codes produced for that batch (2**bsq_bits possible
    ids for bsq, cfg.vocab for softmax).

Note the two quant_type checkpoints differ in Ks (bsq16_ste: Ks=(1,); modes_1_ste/ks41_ste:
Ks=(4,1)) -- not a controlled comparison, just what happened to already exist from prior sessions.

    uv run python scripts/measure_code_entropy.py
    uv run python scripts/measure_code_entropy.py ks41       # substring-filter by checkpoint name
"""
from pathlib import Path

import torch

from qcute import qcute_v5_concat as concat
from qcute import qcute_v5_stack as stack

DEVICE = "cpu"
N_BATCHES = 20
BATCH_SIZE = 16
DATA_PATH = Path("datasets/enwik8_1M.gz")

CHECKPOINTS = [
    ("qcute_v5_concat_bsq16_ste", concat),
    ("qcute_v5_stack_bsq16_ste", stack),
    ("qcute_v5_concat_modes_1_ste", concat),
    ("qcute_v5_stack_ks41_ste", stack),
    ("qcute_v5_stack_ks221_ste", stack),
    ("qcute_v5_stack_ks221_ste_entropyreg", stack),
]


def possible_ids(cfg) -> int:
    return 2 ** cfg.bsq_bits if cfg.quant_type == "bsq" else cfg.vocab


def encode_level(mod, model, seq_repr, i):
    if mod is concat:
        return model.encode_lms[i](seq_repr, level=i, window=model.windows[i], compute_ntp=False)[0]
    return model.encode_lms[i].encode(seq_repr, level=i, window=model.windows[i], compute_ntp=False)[0]


def measure(mod, model, cfg, data, n_batches, batch_size, label):
    n_levels = len(cfg.Ks)
    entropy_sums = [0.0] * n_levels
    id_sets = [set() for _ in range(n_levels)]
    n_positions = [0] * n_levels
    total_entropy_reg = 0.0

    with torch.no_grad():
        for _ in range(n_batches):
            x = mod.sample_context(data, batch_size, cfg.context_len, DEVICE)
            _loss, metrics = model(x)
            total_entropy_reg += float(metrics["entropy_reg_total"])
            for i in range(n_levels):
                key = f"level{i}_entropy_reg"
                if key in metrics:
                    entropy_sums[i] += float(metrics[key])
            seq_repr = x
            for i in range(n_levels):
                c_i = encode_level(mod, model, seq_repr, i)
                ids = model.encode_lms[i].quant.to_ids(c_i).reshape(-1)
                id_sets[i].update(ids.tolist())
                n_positions[i] += ids.numel()
                seq_repr = c_i

    poss = possible_ids(cfg)
    print(f"  [{label}] entropy_reg_total(avg/{n_batches} batches)={total_entropy_reg / n_batches:.4f}")
    for i in range(n_levels):
        used = len(id_sets[i])
        print(f"    level{i}: entropy_reg={entropy_sums[i] / n_batches:.4f}  "
              f"distinct_codes_used={used}/{poss} ({100 * used / poss:.1f}%)  "
              f"positions_seen={n_positions[i]}")


def main():
    import sys
    only = sys.argv[1:]
    checkpoints = [(n, m) for n, m in CHECKPOINTS if not only or any(o in n for o in only)]
    for name, mod in checkpoints:
        ckpt_path = f"checkpoints/{name}/best.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        cfg = mod.Config(**ckpt["cfg"])
        model = mod.RefineLM(cfg).to(DEVICE)
        model.load_state_dict(ckpt["model"])
        model.eval()

        data_all = mod.load_enwik8(DATA_PATH)
        train_data, val_data = mod.split_train_val(data_all, 0.1)

        print(f"=== {name} (module={mod.__name__}, quant_type={cfg.quant_type}, Ks={cfg.Ks}) ===")
        measure(mod, model, cfg, train_data, N_BATCHES, BATCH_SIZE, "TRAIN")
        measure(mod, model, cfg, val_data, N_BATCHES, BATCH_SIZE, "VAL")
        print()


if __name__ == "__main__":
    main()

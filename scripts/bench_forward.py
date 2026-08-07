"""CPU, batch_size=1, single-forward-pass comparison of bytelm (xs preset),
bpelm (vocab=8192), and qcutelm_vlt6 (reads whichever config's
architecture — currently configs/qcutelm_vlt6_rope_bpelm_parity.py) —
params, real FLOP count (torch.utils.flop_counter.FlopCounterMode, not a
d^2*n_layers*tokens proxy), and wallclock. Forced to CPU explicitly so it
never contends with a live MPS training job. See scripts/bench_breakdown.py
for a per-component (encoder/codelm/decode) FLOP breakdown of vlt6 instead
of just the aggregate number this script reports.

    uv run python scripts/bench_forward.py
"""
import sys, time
sys.path.insert(0, '/Users/muaz/code/qcute')
import torch
from torch.utils.flop_counter import FlopCounterMode

device = 'cpu'
torch.manual_seed(0)
B = 1

def bench(name, model, inputs, n_warmup=3, n_time=10):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)
        # FLOP count via torch's FlopCounterMode (single forward)
        with FlopCounterMode(display=False) as fc:
            model(*inputs)
        flops = fc.get_total_flops()

        t0 = time.perf_counter()
        for _ in range(n_time):
            model(*inputs)
        dt = (time.perf_counter() - t0) / n_time
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:32s} params={n_params/1e6:7.3f}M  flops/fwd={flops/1e6:10.2f}M  wallclock/fwd={dt*1000:8.3f}ms")
    return flops, dt

print(f"{'model':32s} {'params':>14s} {'flops/fwd':>16s} {'wallclock/fwd':>16s}  (CPU, batch_size={B})")
print("-"*100)

# --- bytelm (xs preset) ---
from qcute.bytelm import ByteLM, PRESETS as BYTE_PRESETS
cfg_b = BYTE_PRESETS["xs"]
model_b = ByteLM(cfg_b).to(device)
tokens_b = torch.randint(0, cfg_b.vocab, (B, cfg_b.context))
bench("bytelm (xs, context=256)", model_b, (tokens_b,))

# --- bpelm (8192 vocab) ---
from qcute.bpelm import BpeLM, BpeLMConfig
cfg_p = BpeLMConfig(vocab=8192, d_model=256, n_layers=4, n_heads=4, context=256)
model_p = BpeLM(cfg_p).to(device)
tokens_p = torch.randint(0, cfg_p.vocab, (B, cfg_p.context))
bench("bpelm (vocab=8192, context=256)", model_p, (tokens_p,))

# --- current running qcutelm_vlt6 grid config (cell 1: ifsq/ntp/shared) ---
from qcute.archive.qcutelm_vlt6 import ARLatentTokenizer, Config, load_config_module
from pathlib import Path
cfg_kwargs = load_config_module(Path('/Users/muaz/code/qcute/configs/qcutelm_vlt6_rope_bpelm_parity.py'))
cfg_v = Config(
    K=cfg_kwargs['K'], context_len=cfg_kwargs['context_len'], attn_window=cfg_kwargs['attn_window'],
    dq=cfg_kwargs['dq'], quant_type=cfg_kwargs['quant_type'], fsq_levels=cfg_kwargs.get('fsq_levels', 8),
    d_model=cfg_kwargs['d_model'], n_heads=cfg_kwargs['n_heads'], n_layers=cfg_kwargs['n_layers'],
    mlp_mult=cfg_kwargs['mlp_mult'], lm_d_model=cfg_kwargs['lm_d_model'], lm_n_heads=cfg_kwargs['lm_n_heads'],
    lm_n_layers=cfg_kwargs['lm_n_layers'], lm_mlp_mult=cfg_kwargs['lm_mlp_mult'],
    use_rope=cfg_kwargs['use_rope'], use_zero_kv=cfg_kwargs['use_zero_kv'],
    main_ntp_weight=cfg_kwargs['main_ntp_weight'], aux_recon_weight=cfg_kwargs['aux_recon_weight'],
    code_match_weight=cfg_kwargs['code_match_weight'],
)
model_v = ARLatentTokenizer(cfg_v).to(device)
ctx_v = torch.randint(0, 256, (B, cfg_v.context_len))

def vlt6_forward(ctx):
    return model_v(ctx)

# FlopCounterMode + timing for the full forward() (encoder+codelm+decode)
model_v.eval()
with torch.no_grad():
    for _ in range(3):
        vlt6_forward(ctx_v)
    with FlopCounterMode(display=False) as fc:
        vlt6_forward(ctx_v)
    flops_v = fc.get_total_flops()
    t0 = time.perf_counter()
    for _ in range(10):
        vlt6_forward(ctx_v)
    dt_v = (time.perf_counter() - t0) / 10
n_params_v = sum(p.numel() for p in model_v.parameters())
print(f"{'qcutelm_vlt6 (grid cell 1, ctx=1024)':32s} params={n_params_v/1e6:7.3f}M  flops/fwd={flops_v/1e6:10.2f}M  wallclock/fwd={dt_v*1000:8.3f}ms")

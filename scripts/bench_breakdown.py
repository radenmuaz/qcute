"""Per-component FLOP breakdown for qcute.qcutelm_vlt6 (encoder pass,
codelm pass, decode pass — verified to sum to the full forward()'s FLOP
count) against qcute.bpelm as a same-width (d=256) reference point. CPU,
batch_size=1, torch.utils.flop_counter.FlopCounterMode for real FLOP
counts. Explains WHY vlt6's total FLOPs land close to bpelm's despite
having notably fewer total params — see scripts/bench_forward.py for the
higher-level aggregate comparison (bytelm included there too) this script
drills into.

    uv run python scripts/bench_breakdown.py
"""
import sys
sys.path.insert(0, '/Users/muaz/code/qcute')
import torch
from torch.utils.flop_counter import FlopCounterMode
from pathlib import Path

device = 'cpu'
torch.manual_seed(0)
B = 1

def flops_of(fn):
    with torch.no_grad():
        fn()  # warmup
        with FlopCounterMode(display=False) as fc:
            fn()
    return fc.get_total_flops()

# --- bpelm breakdown ---
from qcute.bpelm import BpeLM, BpeLMConfig
cfg_p = BpeLMConfig(vocab=8192, d_model=256, n_layers=4, n_heads=4, context=256)
model_p = BpeLM(cfg_p).to(device).eval()
tokens_p = torch.randint(0, cfg_p.vocab, (B, cfg_p.context))
n_params_p = sum(p.numel() for p in model_p.parameters())
n_params_p_head = model_p.lm_head.weight.numel() if hasattr(model_p, 'lm_head') else None
flops_p_full = flops_of(lambda: model_p(tokens_p))
print("=== bpelm (d=256, n_layers=4, context=256, vocab=8192) ===")
print(f"  total params: {n_params_p/1e6:.3f}M")
print(f"  total flops/fwd: {flops_p_full/1e6:.2f}M")

# --- vlt6 breakdown ---
from qcute.qcutelm_vlt6 import ARLatentTokenizer, Config, load_config_module
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
model_v = ARLatentTokenizer(cfg_v).to(device).eval()
ctx_v = torch.randint(0, 256, (B, cfg_v.context_len))

n_tokenizer = sum(p.numel() for n, p in model_v.named_parameters() if not n.startswith('codelm'))
n_codelm = sum(p.numel() for n, p in model_v.named_parameters() if n.startswith('codelm'))
print()
print(f"=== qcutelm_vlt6 (tokenizer d={cfg_v.d_model} n_layers={cfg_v.n_layers}, "
      f"codelm d={cfg_v.lm_d_model} n_layers={cfg_v.lm_n_layers}, context_len={cfg_v.context_len}) ===")
print(f"  tokenizer (enc+dec, shared) params: {n_tokenizer/1e6:.3f}M")
print(f"  codelm params:                      {n_codelm/1e6:.3f}M")
print(f"  total params:                       {(n_tokenizer+n_codelm)/1e6:.3f}M")

with torch.no_grad():
    h = model_v.run_blocks(model_v.byte_emb(ctx_v))
flops_encoder = flops_of(lambda: model_v.run_blocks(model_v.byte_emb(ctx_v)))

z_hat = model_v.codes_from_hidden(h)
code_embed_true = model_v.z_proj(z_hat)
flops_codelm = flops_of(lambda: model_v.codelm(code_embed_true))

n_blocks = cfg_v.context_len // cfg_v.K
target_blocks = ctx_v.view(B, n_blocks, cfg_v.K)[:, 1:, :]
pred_soft_full, _ = model_v.codelm(code_embed_true)
pred_soft = pred_soft_full[:, :-1, :]
pred_soft_flat = pred_soft.reshape(B * (n_blocks - 1), cfg_v.dq)
target_flat = target_blocks.reshape(B * (n_blocks - 1), cfg_v.K)
flops_decode = flops_of(lambda: model_v.decode_block(pred_soft_flat, target_flat))

flops_v_full = flops_of(lambda: model_v(ctx_v))

print(f"  flops: encoder pass (ctx={cfg_v.context_len} bytes):  {flops_encoder/1e6:8.2f}M")
print(f"  flops: codelm pass (codes={n_blocks}):               {flops_codelm/1e6:8.2f}M")
print(f"  flops: decode pass ({n_blocks-1} blocks x {cfg_v.K} bytes): {flops_decode/1e6:8.2f}M")
print(f"  flops: sum of parts:                                {(flops_encoder+flops_codelm+flops_decode)/1e6:8.2f}M")
print(f"  flops: full forward() (measured directly):          {flops_v_full/1e6:8.2f}M")

print()
print("=== comparison: vlt6's codelm alone vs bpelm (both d=256) ===")
print(f"  codelm:  d=256, n_layers={cfg_v.lm_n_layers}, tokens={n_blocks} (codes) -> {flops_codelm/1e6:.2f}M flops, {n_codelm/1e6:.3f}M params")
print(f"  bpelm:   d=256, n_layers={cfg_p.n_layers}, tokens={cfg_p.context} (bytes) -> {flops_p_full/1e6:.2f}M flops, {n_params_p/1e6:.3f}M params")

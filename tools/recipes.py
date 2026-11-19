#!/usr/bin/env python3
"""Memory model and TabbyAPI configs for the SenseNova-Loop-9B EXL3 quants.

    python tools/recipes.py                       # print the model, write configs/*.yml and print the recipe table
    python tools/recipes.py --card-mib 16376 --branch 5.0 --kv q8 --reserve 1.75 --parallel 1   # one custom recipe

Model (fitted on data/vram-sweeps, one request, BF16 vision tower loaded, recurrent state FP32):

    GPU memory (GiB) = base[branch, kv] + per_token[kv] * cache_size + 0.2 (prefill chunk 2048) + 0.167 * (parallel - 1)

base includes the weights, vision tower, buffers and 470 MiB of non-PyTorch memory (CUDA context and kernels, measured
with nvidia-smi on an idle RTX PRO 4500). The reserve covers what the model does not: the desktop/display, allocator and
library slack, and the vision tower's working memory for one image (0.106 GiB per megapixel, data/vision-mem.json).
"""
import argparse, glob, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NON_TORCH_MIB = 470
CHUNK_GIB = 0.2          # chunk_size 2048 vs the 256 used in the sweeps (measured +0.17..0.21 GiB)
PER_REQUEST_GIB = 0.167  # data/batchmem-5.0.json: 1 -> 4 -> 8 parallel requests, same pool
NATIVE_MAX = 524288      # YaRN factor 2 over 262,144 in the model config
CACHES = ['fp16', 'q8', 'q6', 'q4']
# nvidia-smi "total" of typical cards (MiB): 12 GB RTX 3060/4070, 16 GB 4060 Ti/4080/5070 Ti, 24 GB 3090/4090, 32 GB 5090
CARDS = {12: 12282, 16: 16376, 24: 24564, 32: 32607}
RECIPES = {12: '4.0', 16: '5.0', 24: '6.0', 32: '8.0'}
RESERVE = {'display': 1.0, 'slack': 0.3, 'image_4mp': 0.45}


def peak_gib(row):
    v = row.get('proc_peak_mib')
    if v is None and row.get('torch_max_allocated_bytes') is not None:
        v = row['torch_max_allocated_bytes'] / 2**20 + NON_TORCH_MIB
    return None if v is None else v / 1024


def load():
    data = {}
    for p in sorted(glob.glob(str(ROOT / 'data' / 'vram-sweeps' / 'vram-*.json'))):
        m = re.match(r'vram-(\d\.\d)(\.partial)?\.json$', Path(p).name)
        rows = [r for r in json.loads(Path(p).read_text())['rows'] if r['status'] == 'ok' and peak_gib(r) is not None]
        if m and rows: data[m.group(1)] = rows
    return data


def fit(data):
    full = {q: rows for q, rows in data.items() if len(rows) >= 8}
    slope = {}
    for c in CACHES:
        num = den = 0.0
        for rows in full.values():
            pts = [(r['context'], peak_gib(r)) for r in rows if r['cache'] == c]
            if len(pts) < 2: continue
            mx = sum(x for x, _ in pts) / len(pts); my = sum(y for _, y in pts) / len(pts)
            num += sum((x - mx) * (y - my) for x, y in pts); den += sum((x - mx) ** 2 for x, _ in pts)
        slope[c] = num / den
    base = {}
    for q, rows in data.items():
        base[q] = {c: sum(y - slope[c] * x for x, y in pts) / len(pts)
                   for c in CACHES for pts in [[(r['context'], peak_gib(r)) for r in rows if r['cache'] == c]] if pts}
        ref = next(iter(base[q].values()))
        for c in CACHES: base[q].setdefault(c, ref)
    return slope, base


def pool_tokens(slope, base, branch, kv, card_mib, reserve, parallel=1, step=4096):
    free = card_mib / 1024 - reserve - base[branch][kv] - CHUNK_GIB - PER_REQUEST_GIB * (parallel - 1)
    return int(free / slope[kv] // step * step)


def config(branch, kv, cache_size, parallel):
    seq = min(cache_size, NATIVE_MAX)
    return f"""model:
  model_name: SenseNova-Loop-9B-exl3-{branch}bpw
  max_seq_len: {seq}
  cache_size: {cache_size}
  cache_mode: {kv.upper()}
  chunk_size: 2048
  max_batch_size: {parallel}
  recurrent_state_fp32: true
  warmup: true               # allocate kernels/workspaces at load, not during the first request
  tool_format: qwen3_coder
  vision: true
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--card-mib', type=int); p.add_argument('--branch'); p.add_argument('--kv', default='q8')
    p.add_argument('--reserve', type=float, default=sum(RESERVE.values())); p.add_argument('--parallel', type=int, default=1)
    a = p.parse_args()
    slope, base = fit(load())
    if a.card_mib:
        n = pool_tokens(slope, base, a.branch, a.kv, a.card_mib, a.reserve, a.parallel)
        print(config(a.branch, a.kv, n, a.parallel)); return
    print('GiB per 1k tokens: ' + ', '.join(f'{c} {slope[c] * 1024:.4f}' for c in CACHES))
    print('base GiB (Q8 KV): ' + ', '.join(f'{q} {base[q]["q8"]:.2f}' for q in sorted(base, reverse=True)))
    out = ROOT / 'configs'; out.mkdir(exist_ok=True)
    print('\n| Card | Branch | parallel | cache_size | max_seq_len | model + pool GiB | card GiB |')
    for card, branch in RECIPES.items():
        for par in (1, 4):
            n = pool_tokens(slope, base, branch, 'q8', CARDS[card], sum(RESERVE.values()), par)
            used = base[branch]['q8'] + CHUNK_GIB + PER_REQUEST_GIB * (par - 1) + slope['q8'] * n
            name = f'tabby-{card}gb' + ('' if par == 1 else f'-{par}x') + '.yml'
            (out / name).write_text(f'# {card} GB card: {branch}bpw, Q8 KV, {par} request(s) at a time; generated by tools/recipes.py\n'
                                    + config(branch, 'q8', n, par))
            print(f'| {card} GB | {branch}bpw | {par} | {n} | {min(n, NATIVE_MAX)} | {used:.2f} | {CARDS[card] / 1024:.2f} |')


if __name__ == '__main__':
    main()

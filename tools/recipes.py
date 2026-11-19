#!/usr/bin/env python3
"""Memory model and TabbyAPI configs for the SenseNova-Loop-9B EXL3 quants.

    python tools/recipes.py --card 24                      # config for a 24 GB card used only for the model
    python tools/recipes.py --card 24 --desktop            # the card also drives your display (keeps 1 GiB free)
    python tools/recipes.py --card 16 --no-vision          # text only: no vision tower, more context
    python tools/recipes.py --card-mib 20475 --branch 6.0 --kv fp16 --parallel 2 --image-mp 8
    python tools/recipes.py --write-configs                # regenerate configs/*.yml and print the table

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
DISPLAY_GIB = 1.0       # desktop/compositor share when the card drives a display (--desktop)
SLACK_GIB = 0.3         # PyTorch allocator and library slack
VISION_TOWER_GIB = 0.932  # BF16 tower, included in base; removed with --no-vision (data/vision-mem.json)
IMAGE_GIB_PER_MP = 0.111  # vision tower working memory per MP while encoding one image: the 4 MP point of data/vision-mem.json (0.444 GiB), conservative vs the 0.106 slope


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


def reserve_gib(desktop=False, vision=True, image_mp=4.0):
    return SLACK_GIB + (DISPLAY_GIB if desktop else 0.0) + (IMAGE_GIB_PER_MP * image_mp if vision else 0.0)


def pool_tokens(slope, base, branch, kv, card_mib, reserve, parallel=1, vision=True, step=4096):
    b = base[branch][kv] - (0.0 if vision else VISION_TOWER_GIB)
    free = card_mib / 1024 - reserve - b - CHUNK_GIB - PER_REQUEST_GIB * (parallel - 1)
    return int(free / slope[kv] // step * step)


def config(branch, kv, cache_size, parallel, vision=True):
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
  vision: {"true" if vision else "false"}
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--card', type=int, choices=sorted(CARDS), help='card class in GB (typical nvidia-smi total)')
    p.add_argument('--card-mib', type=int, help='exact total from: nvidia-smi --query-gpu=memory.total --format=csv')
    p.add_argument('--branch', choices=['8.0', '6.0', '5.0', '4.0'], help='model version (default: the recipe for the card class)')
    p.add_argument('--kv', choices=CACHES, default='q8', help='KV cache type (default q8: no measurable quality cost)')
    p.add_argument('--desktop', action='store_true', help='the card also drives your display: keep 1 GiB free for it')
    p.add_argument('--no-vision', action='store_true', help='text only: do not load the vision tower')
    p.add_argument('--image-mp', type=float, default=4.0, help='largest image you will send, in megapixels (default 4)')
    p.add_argument('--parallel', type=int, default=1, help='requests served at the same time (default 1)')
    p.add_argument('--write-configs', action='store_true', help='regenerate configs/*.yml and print the table')
    a = p.parse_args()
    slope, base = fit(load())
    if not a.write_configs:
        if not (a.card or a.card_mib): p.error('give --card or --card-mib (or --write-configs)')
        mib = a.card_mib or CARDS[a.card]
        branch = a.branch or RECIPES.get(a.card) or max((b for c, b in RECIPES.items() if CARDS[c] <= mib), default='4.0')
        vision = not a.no_vision
        n = pool_tokens(slope, base, branch, a.kv, mib, reserve_gib(a.desktop, vision, a.image_mp), a.parallel, vision)
        if n < 4096: p.error('does not fit: choose a smaller --branch or --kv')
        print(f'# {mib} MiB card, {branch}bpw, {a.kv.upper()} KV, {"desktop" if a.desktop else "dedicated GPU"}, '
              f'{"vision up to %g MP" % a.image_mp if vision else "text only"}, {a.parallel} request(s); generated by tools/recipes.py')
        print(config(branch, a.kv, n, a.parallel, vision), end=''); return
    out = ROOT / 'configs'; out.mkdir(exist_ok=True)
    for f in out.glob('tabby-*.yml'): f.unlink()
    print('| Card | Branch | desktop | parallel | cache_size | max_seq_len | GiB used | card GiB |')
    for card, branch in RECIPES.items():
        for desktop in (False, True):
            for par in (1, 4):
                n = pool_tokens(slope, base, branch, 'q8', CARDS[card], reserve_gib(desktop), par)
                used = base[branch]['q8'] + CHUNK_GIB + PER_REQUEST_GIB * (par - 1) + slope['q8'] * n
                name = f'tabby-{card}gb' + ('-desktop' if desktop else '') + ('' if par == 1 else f'-{par}x') + '.yml'
                (out / name).write_text(f'# {card} GB card, {"also driving the display" if desktop else "used only for the model"}: '
                                        f'{branch}bpw, Q8 KV, {par} request(s) at a time; generated by tools/recipes.py\n'
                                        + config(branch, 'q8', n, par))
                print(f'| {card} GB | {branch}bpw | {desktop} | {par} | {n} | {min(n, NATIVE_MAX)} | {used:.2f} | {CARDS[card] / 1024:.2f} |')


if __name__ == '__main__':
    main()

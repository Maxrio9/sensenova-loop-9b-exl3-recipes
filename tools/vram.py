#!/usr/bin/env python3
"""Runtime VRAM sweep for the Loop EXL3 quants: (cache type x context), one fresh child process each.

Per row it records torch max allocated/reserved, nvidia-smi peaks (whole device and this process), weights/cache
bytes, load-only VRAM, the optional BF16 vision tower VRAM and one image answer, and success/OOM.

CAP MODE (--cap-gib X) emulates a card of X GiB on a bigger GPU. What it does:
  * torch.cuda.set_per_process_memory_fraction so that torch's caching allocator (allocated+cached) can use at most
    X - headroom - measured non-torch overhead (CUDA context, kernel modules; measured after init as
    per-process nvidia-smi minus torch reserved, plus --overhead-margin-gib for lazy loads). Exceeding it raises a
    CUDA OOM, recorded as status=oom.
  * after the run it checks that the per-process nvidia-smi peak is <= X - headroom (cap_pass).
What it does NOT do: it does not limit non-torch allocations made after the overhead was measured (cuBLAS/Triton/
extension workspaces, pinned memory), does not emulate the smaller card's bandwidth/clocks, and other processes on
the device are not capped. The post-run nvidia-smi check is what catches overhead growth.
Per-process memory: nvidia-smi --query-compute-apps matched by pid; in a container pids differ, so the fallback is
(sum of all compute apps now) - (sum before this process started), reported as proc_source.
"""
import argparse, json, os, subprocess, sys, threading

MARK = 'VRAMRESULT '
CACHE_BITS = {'fp16': None, 'q8': 8, 'q6': 6, 'q4': 4}
DEFAULT_CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_CACHES = ['fp16', 'q8', 'q6', 'q4']
MEDIA = ['exl3/exllamav3/examples/media/cat.png']


def parse_compute_apps(text, pid):
    """-> (used_mib of `pid` or None, total used_mib of all apps) from `--query-compute-apps=pid,used_memory`."""
    mine, total = None, 0
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) < 2: continue
        try: p, m = int(parts[0]), int(parts[1])
        except ValueError: continue
        total += m
        if p == pid: mine = (mine or 0) + m
    return mine, total


def smi_snapshot(pid):
    """Device used MiB (max over GPUs listed) plus per-process numbers; Nones if nvidia-smi fails."""
    try:
        dev = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True, timeout=5)
        mine, total = parse_compute_apps(apps, pid)
        return {'device': max(int(x.strip()) for x in dev.splitlines()), 'mine': mine, 'apps_total': total}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def proc_mib(snap, base):
    """Per-process used MiB and the source of the number."""
    if snap is None: return None, 'unavailable'
    if snap['mine'] is not None: return snap['mine'], 'pid'
    return max(0, snap['apps_total'] - (base['apps_total'] if base else 0)), 'delta'


def evaluate_cap(row, cap_gib, headroom_gib):
    """Post-run verification: per-process nvidia-smi peak must stay within cap - headroom."""
    limit = cap_gib - headroom_gib
    peak = row.get('proc_peak_mib')
    if peak is None and row.get('torch_max_allocated_bytes') is not None:
        # No per-process nvidia-smi (DGX Spark/GB10): torch peak + the non-torch overhead measured on the RTX PRO 4500
        # (455-473 MiB, logs/perf-6.0.json), the same estimate vram_table.py uses; the run itself was torch-capped.
        peak = row['torch_max_allocated_bytes'] / 2**20 + 470
        row['cap_peak_source'] = 'torch+470MiB'
    return row.get('status') == 'ok' and peak is not None and peak / 1024 <= limit


def torch_fraction(cap_gib, headroom_gib, overhead_mib, total_mib):
    """Allocator fraction so torch memory + non-torch overhead <= cap - headroom."""
    budget = (cap_gib - headroom_gib) * 1024 - overhead_mib
    if budget <= 0: raise ValueError(f'overhead {overhead_mib:.0f} MiB leaves no budget in a {cap_gib} GiB cap')
    return min(1.0, budget / total_mib)


def find_image(path):
    here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(os.path.dirname(here))
    cands = ([path] if path else []) + [os.path.join(root, m) for m in MEDIA]
    try:
        import exllamav3
        cands.append(os.path.join(os.path.dirname(os.path.dirname(exllamav3.__file__)), 'examples', 'media', 'cat.png'))
    except ImportError: pass
    for c in cands:
        if c and os.path.exists(c): return c
    raise FileNotFoundError('no test image found: ' + ', '.join(map(str, cands)))


def child(a):
    pid = os.getpid(); base = smi_snapshot(pid)
    res = {'cache': a.cache, 'context': a.context, 'vision': a.vision, 'cap_gib': a.cap_gib, 'status': 'fail', 'error': None,
           'baseline_device_mib': base and base['device'], 'baseline_apps_mib': base and base['apps_total']}
    stop = threading.Event(); peaks = {'device': 0, 'proc': 0, 'source': 'unavailable'}
    def poll():
        while not stop.is_set():
            s = smi_snapshot(pid)
            if s:
                peaks['device'] = max(peaks['device'], s['device'])
                v, src = proc_mib(s, base); peaks['proc'] = max(peaks['proc'], v); peaks['source'] = src
            stop.wait(.1)
    th = threading.Thread(target=poll, daemon=True); th.start()
    torch = None
    try:
        import torch
        torch.set_grad_enabled(False)
        from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler, CacheLayer_quant
        dev_props = torch.cuda.get_device_properties(0); total_mib = dev_props.total_memory / 2**20
        res['device_name'] = dev_props.name; res['device_total_mib'] = total_mib
        a_ = torch.randn(256, 256, device='cuda'); (a_ @ a_).sum().item(); del a_   # init context + cuBLAS
        torch.cuda.synchronize()
        s = smi_snapshot(pid); pm, src = proc_mib(s, base)
        overhead = (pm - torch.cuda.memory_reserved() / 2**20) if pm is not None else 0.0
        res['overhead_mib'] = overhead; res['proc_source'] = src
        if a.cap_gib:
            frac = torch_fraction(a.cap_gib, a.headroom_gib, overhead + a.overhead_margin_gib * 1024, total_mib)
            torch.cuda.set_per_process_memory_fraction(frac, 0); res['cap_fraction'] = frac
        mib = lambda: torch.cuda.memory_allocated() / 2**20
        cfg = Config.from_directory(a.model)
        vision = None
        if a.vision:
            before_a = torch.cuda.memory_allocated(); sm = smi_snapshot(pid); before_p = proc_mib(sm, base)[0] or 0
            vision = Model.from_config(cfg, component='vision'); vision.load(progressbar=False)
            torch.cuda.synchronize(); sm = smi_snapshot(pid)
            res['vision_torch_bytes'] = torch.cuda.memory_allocated() - before_a
            res['vision_proc_mib'] = (proc_mib(sm, base)[0] or 0) - before_p
        model = Model.from_config(cfg)
        kw = {} if CACHE_BITS[a.cache] is None else {'layer_type': CacheLayer_quant, 'k_bits': CACHE_BITS[a.cache], 'v_bits': CACHE_BITS[a.cache]}
        tokens = (a.context + 512 + 255) // 256 * 256
        cache = Cache(model, max_num_tokens=tokens, max_batch_size=1, recurrent_dtype=torch.float, **kw)
        before_a = torch.cuda.memory_allocated()
        model.load(progressbar=False, max_chunk_size=256, max_output_size=256)
        torch.cuda.synchronize()
        text_bytes = torch.cuda.memory_allocated() - before_a
        cache_bytes = sum(l.storage_size() for l in cache.layers.values())
        res['cache_bytes'] = cache_bytes; res['cache_tokens'] = tokens
        res['text_load_alloc_bytes'] = text_bytes
        res['weights_bytes'] = text_bytes - cache_bytes   # includes recurrent state pool and any loader workspace still held
        sm = smi_snapshot(pid); res['load_only_proc_mib'] = proc_mib(sm, base)[0]; res['load_only_device_mib'] = sm and sm['device']
        tok = Tokenizer.from_config(cfg); gen = Generator(model, cache, tok, max_batch_size=1, max_chunk_size=256)

        def run(ids, new_tokens, **jkw):
            job = Job(input_ids=ids, max_new_tokens=new_tokens, sampler=GreedySampler(), identifier=0, decode_special_tokens=True,
                      stop_conditions=[tok.eos_token_id, '<|im_end|>'] if jkw else [], **jkw)
            gen.enqueue(job); out = ''; n = 0
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    out += r.get('text', '')
                    if r.get('eos'): n = r['new_tokens']
            return out, n
        torch.cuda.reset_peak_memory_stats()
        text = 'Benchmark sequence. ' + 'Science and technology have many applications. ' * (a.context // 6 + 10)
        ids = tok.encode(text)[:, :a.context]
        if ids.shape[-1] != a.context: raise ValueError('workload too short')
        _, n = run(ids, 64)
        if n != 64: raise RuntimeError(f'text request generated {n} tokens, expected 64')
        res['text_request_ok'] = True
        if a.vision:
            from PIL import Image
            emb = vision.get_image_embeddings(tokenizer=tok, image=Image.open(find_image(a.image)))
            prompt = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n' + emb.text_alias +
                      '\nWhat is in this image? Answer briefly.<|im_end|>\n<|im_start|>assistant\n')
            ids = tok.encode(prompt, encode_special_tokens=True, embeddings=[emb])
            answer, _ = run(ids, 64, embeddings=[emb])
            res['image_answer'] = answer; res['image_request_ok'] = bool(answer.strip())
        torch.cuda.synchronize()
        res['status'] = 'ok'
    except BaseException as e:
        msg = f'{type(e).__name__}: {e}'
        res['error'] = msg[-500:]
        if 'out of memory' in msg.lower() or (torch is not None and isinstance(e, getattr(torch, 'OutOfMemoryError', ()))):
            res['status'] = 'oom'
        if isinstance(e, KeyboardInterrupt): raise
    finally:
        stop.set(); th.join(timeout=3)
        if torch is not None and torch.cuda.is_available():
            res['torch_max_allocated_bytes'] = torch.cuda.max_memory_allocated()
            res['torch_max_reserved_bytes'] = torch.cuda.max_memory_reserved()
        res['device_peak_mib'] = peaks['device'] or None; res['proc_peak_mib'] = peaks['proc'] or None
        res['proc_source'] = peaks['source'] if peaks['source'] != 'unavailable' else res.get('proc_source')
        if a.cap_gib: res['cap_pass'] = evaluate_cap(res, a.cap_gib, a.headroom_gib)
        print(MARK + json.dumps(res, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), flush=True)


def parse_pairs(s):
    out = []
    for item in s.split(','):
        c, n = item.split(':'); out.append((c, int(n)))
    for c, _ in out:
        if c not in CACHE_BITS: raise argparse.ArgumentTypeError(f'unknown cache type {c}')
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True); p.add_argument('--json', default='vram.json')
    p.add_argument('--contexts', nargs='+', type=int, default=DEFAULT_CONTEXTS)
    p.add_argument('--caches', nargs='+', choices=list(CACHE_BITS), default=DEFAULT_CACHES)
    p.add_argument('--pairs', type=parse_pairs, help='explicit CACHE:CONTEXT,... list (overrides --caches/--contexts), e.g. fp16:32768,q4:131072')
    p.add_argument('--vision', action='store_true', help='also load the BF16 vision tower and run one image prompt')
    p.add_argument('--image', help='test image (default: examples/media/cat.png of the fork)')
    p.add_argument('--cap-gib', type=float, help='emulate a card of this many GiB (see module doc)')
    p.add_argument('--headroom-gib', type=float, default=0.75)
    p.add_argument('--overhead-margin-gib', type=float, default=0.25, help='extra non-torch allowance on top of the measured overhead (cap mode)')
    p.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--cache', choices=list(CACHE_BITS), help=argparse.SUPPRESS); p.add_argument('--context', type=int, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def build_child_cmd(a, cache, context):
    cmd = [sys.executable, os.path.abspath(__file__), '--model', a.model, '--child', '--cache', cache, '--context', str(context),
           '--headroom-gib', str(a.headroom_gib), '--overhead-margin-gib', str(a.overhead_margin_gib)]
    if a.vision: cmd.append('--vision')
    if a.image: cmd += ['--image', a.image]
    if a.cap_gib: cmd += ['--cap-gib', str(a.cap_gib)]
    return cmd


def run_one(a, cache, context, runner=subprocess.run):
    cp = runner(build_child_cmd(a, cache, context), capture_output=True, text=True)
    match = [x[len(MARK):] for x in cp.stdout.splitlines() if x.startswith(MARK)]
    if match:
        row = json.loads(match[-1])
        if cp.returncode != 0 and row['status'] == 'ok': row['status'] = 'fail'; row['error'] = f'child rc={cp.returncode}'
        return row
    status = 'oom' if 'out of memory' in (cp.stdout + cp.stderr).lower() else 'fail'
    return {'cache': cache, 'context': context, 'vision': a.vision, 'cap_gib': a.cap_gib, 'status': status,
            'error': f'child rc={cp.returncode}, no result: ' + (cp.stderr or cp.stdout)[-500:]}


def main(argv=None, runner=subprocess.run):
    a = parse_args(argv)
    if a.child: child(a); return 0
    pairs = a.pairs or [(c, n) for c in a.caches for n in a.contexts]
    rows = []
    for cache, ctx in pairs:
        row = run_one(a, cache, ctx, runner)
        rows.append(row)
        tag = {'ok': 'PASS', 'oom': 'OOM '}.get(row['status'], 'FAIL')
        cap = '' if not a.cap_gib else f" cap_pass={row.get('cap_pass')}"
        print(f"{tag} vram cache={cache} context={ctx} proc_peak_mib={row.get('proc_peak_mib')} device_peak_mib={row.get('device_peak_mib')}{cap}", flush=True)
        write(a, rows, partial=True)
    write(a, rows)
    # sweep: any non-OOM failure is an error; cap mode: any failed verification is an error
    bad = any(r['status'] == 'fail' for r in rows) or not rows
    if a.cap_gib: bad = bad or any(not r.get('cap_pass') for r in rows)
    return int(bad)


def write(a, rows, partial=False):
    meta = {'model': a.model, 'cap_gib': a.cap_gib, 'headroom_gib': a.headroom_gib, 'vision': a.vision,
            'cap_enforcement': 'torch allocator fraction = (cap - headroom - overhead - margin); nvidia-smi per-process peak verified afterwards; non-torch growth after init is not enforced',
            'overhead_margin_gib': a.overhead_margin_gib}
    with open(a.json + ('.partial' if partial else ''), 'w') as f: json.dump({'meta': meta, 'rows': rows}, f, indent=2)


if __name__ == '__main__':
    raise SystemExit(main())

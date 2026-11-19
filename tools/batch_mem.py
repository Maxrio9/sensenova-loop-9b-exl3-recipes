#!/usr/bin/env python3
"""Per-request memory overhead: same KV pool, max_batch_size 1/4/8, that many parallel requests (2k prompt + 32 tokens each).
Peak PyTorch allocation per setting; the slope over batch size is the cost of one more concurrent request."""
import argparse, json, subprocess, sys
import torch
torch.set_grad_enabled(False)


def child(model_dir, batch, pool):
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler, CacheLayer_quant
    cfg = Config.from_directory(model_dir)
    vision = Model.from_config(cfg, component='vision'); vision.load(progressbar=False)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=pool, max_batch_size=batch, recurrent_dtype=torch.float,
                  layer_type=CacheLayer_quant, k_bits=8, v_bits=8)
    model.load(progressbar=False, max_chunk_size=2048, max_output_size=256)
    tok = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tok, max_batch_size=batch, max_chunk_size=2048)
    for i in range(batch):
        ids = tok.encode(f'Request {i}. ' + 'The river bends past the old mill and the orchard. ' * 200)[:, :2048]
        gen.enqueue(Job(input_ids=ids, max_new_tokens=32, sampler=GreedySampler(), identifier=i))
    n = 0
    while gen.num_remaining_jobs():
        n += sum(1 for r in gen.iterate() if r.get('eos'))
    torch.cuda.synchronize()
    print('RESULT ' + json.dumps({'batch': batch, 'pool': pool, 'finished': n, 'peak': torch.cuda.max_memory_allocated()}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--pool', type=int, default=65536)
    p.add_argument('--child', type=int); p.add_argument('--out', required=True); a = p.parse_args()
    if a.child: child(a.model, a.child, a.pool); return
    rows = []
    for b in (1, 4, 8):
        out = subprocess.run([sys.executable, __file__, '--model', a.model, '--pool', str(a.pool), '--child', str(b), '--out', a.out],
                             capture_output=True, text=True).stdout
        rows += [json.loads(l[7:]) for l in out.splitlines() if l.startswith('RESULT ')]
        print(rows[-1] if rows else out[-2000:], flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)
    if len(rows) == 3:
        per = (rows[2]['peak'] - rows[0]['peak']) / 7 / 2**30
        print(f'PER-REQUEST overhead {per:.3f} GiB (batch 1 -> 8, same pool)', flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Long-context retrieval + memory check for one EXL3 quant with FP16/Q8/Q6 KV caches (vision tower loaded).

Haystack: the model's own generations from the training pool (prose categories, tool calls excluded), decoded to text
and laid out as numbered documents. Five ordinary facts about invented places/people are added as paragraphs inside
documents at 10/30/50/70/90% depth. No codes, secrets or passwords (the model refuses such lookups), and the entities
are invented so the answers cannot come from training data. One prompt per (context, KV type) asks all five questions;
thinking off, greedy. Each answer line is graded by its expected value. A 4k control run checks format/refusals.

  long_ctx.py --model DIR --out results.jsonl [--runs 4096:fp16,131072:fp16,131072:q8,...]
"""
import argparse, glob, json, os, random, re, time
import torch
torch.set_grad_enabled(False)

FACTS = [
    ('The tool shed at the community garden on Harrow Lane opens at 6:45 on Saturday mornings.',
     'At what time does the tool shed at the community garden on Harrow Lane open on Saturday mornings?', [r'6[:.]45']),
    ('When the footbridge over the Velka river near Ostrin was rebuilt in 1987, its limestone came from the Kaldor quarry.',
     'Which quarry supplied the limestone when the footbridge over the Velka river near Ostrin was rebuilt?', [r'kaldor']),
    ('Mirela Vos won the final round of the Brennick chess club spring tournament playing the Caro-Kann Defence.',
     'Which opening did Mirela Vos play to win the final round of the Brennick chess club spring tournament?', [r'caro[- ]?kann']),
    ('The research vessel Halcyon Drift sails with a crew of 23 people.',
     'How many crew members sail on the research vessel Halcyon Drift?', [r'\b23\b', r'twenty[- ]three']),
    ("Aunt Ilse's plum cake recipe from the village of Merrow calls for 340 grams of flour.",
     "How many grams of flour does Aunt Ilse's plum cake recipe from Merrow call for?", [r'\b340\b']),
]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
PROSE = {'general', 'reasoning', 'long_context', 'math', 'german'}


def documents(tok_hf, pool_glob, seed=7):
    docs = []
    for f in sorted(glob.glob(pool_glob)):
        for line in open(f):
            r = json.loads(line)
            if r.get('category') not in PROSE: continue
            text = tok_hf.decode(r['token_ids'][r['prompt_tokens']:], skip_special_tokens=True)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()   # answers only
            if len(text) > 400: docs.append(text)
        if len(docs) > 4000: break
    random.Random(seed).shuffle(docs)
    return docs


def build_prompt(tok, docs, length):
    """Fill documents until the prompt is ~length tokens; insert facts at fixed depths (by document index)."""
    head = '<|im_start|>user\nBelow is a collection of documents. Read them carefully; questions follow at the end.\n\n'
    qs = '\n'.join(f'{i + 1}. {q}' for i, (_, q, _) in enumerate(FACTS))
    tail = ('\n\nAnswer the following questions using only the documents above. Give one short answer per line, '
            'numbered 1 to 5.\n' + qs + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    budget = length - len(tok.encode(head + tail, encode_special_tokens=True)[0]) - 300
    if length < 32768:   # short control: trimmed documents, so each fact still lands in its own document
        docs = [d[:1200] for d in docs]
    chosen, used = [], 0
    for d in docs:
        n = len(tok.encode(d)[0]) + 8
        if used + n > budget: break
        chosen.append(d); used += n
    if len(chosen) < len(FACTS): raise ValueError('not enough documents')
    body = [f'[Document {i + 1}]\n{d}' for i, d in enumerate(chosen)]
    for (fact, _, _), depth in zip(FACTS, DEPTHS):
        k = min(len(body) - 1, int(depth * len(body)))
        paras = body[k].split('\n\n')
        paras.insert(max(1, len(paras) // 2), fact)
        body[k] = '\n\n'.join(paras)
    return head + '\n\n'.join(body) + tail


def grade(answer):
    lines = {}
    for line in answer.splitlines():
        m = re.match(r'\s*(\d)[.)]\s*(.*)', line)
        if m: lines.setdefault(int(m.group(1)), m.group(2))
    out = []
    for i, (_, _, pats) in enumerate(FACTS):
        a = lines.get(i + 1, '')
        out.append(bool(a) and any(re.search(p, a, re.I) for p in pats))
    refusal = bool(re.search(r"\b(can't|cannot|unable to|not able to|won't)\b", answer, re.I))
    return out, refusal


def child(a, length, kv):
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, GreedySampler, CacheLayer_quant
    docs = json.load(open(a.docs))
    cfg = Config.from_directory(a.model)
    vision = Model.from_config(cfg, component='vision'); vision.load(progressbar=False)   # card footprint includes the tower
    model = Model.from_config(cfg)
    tok = Tokenizer.from_config(cfg)
    prompt = build_prompt(tok, docs, length)
    ids = tok.encode(prompt, encode_special_tokens=True)
    kwargs = {} if kv == 'fp16' else dict(layer_type=CacheLayer_quant, k_bits=int(kv[1:]), v_bits=int(kv[1:]))
    cache = Cache(model, max_num_tokens=(ids.shape[-1] + 512 + 255) // 256 * 256, max_batch_size=1,
                  recurrent_dtype=torch.float, **kwargs)
    model.load(progressbar=False, max_chunk_size=2048, max_output_size=256)
    gen = Generator(model, cache, tok, max_batch_size=1, max_chunk_size=2048)
    t0 = time.time()
    answer = gen.generate(prompt=prompt, max_new_tokens=120, sampler=GreedySampler(), encode_special_tokens=True,
                          stop_conditions=[tok.eos_token_id, '<|im_end|>'], add_bos=False, completion_only=True)
    secs = time.time() - t0
    ok, refusal = grade(answer)
    peak = torch.cuda.max_memory_allocated()
    rec = dict(context=length, kv=kv, prompt_tokens=int(ids.shape[-1]), correct=sum(ok), per_fact=ok, refusal=refusal,
               answer=answer, seconds=round(secs, 1), torch_max_allocated_bytes=int(peak),
               card_estimate_gib=round(peak / 2**30 + 470 / 1024, 2))
    with open(a.out, 'a') as f: f.write(json.dumps(rec) + '\n')
    print(f"RUN {length // 1024}k {kv}: {sum(ok)}/5 refusal={refusal} {secs:.0f}s card~{rec['card_estimate_gib']} GiB | {answer.strip()[:200]!r}", flush=True)


def main():
    import subprocess, sys
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--out', required=True)
    p.add_argument('--pool', default='/data/train-pool/shard-0000[0-2].jsonl')
    p.add_argument('--docs', default=None, help='haystack documents JSON (built once from --pool)')
    p.add_argument('--child', default=None)
    p.add_argument('--runs', default='4096:fp16,131072:fp16,131072:q8,131072:q6,262144:fp16,262144:q8,262144:q6,'
                                     '393216:q8,393216:q6,524288:q8,524288:q6')
    a = p.parse_args()
    a.docs = a.docs or os.path.splitext(a.out)[0] + '.docs.json'
    if a.child:
        length, kv = a.child.split(':'); child(a, int(length), kv); return
    if not os.path.exists(a.docs):
        from transformers import AutoTokenizer
        docs = documents(AutoTokenizer.from_pretrained(a.model, local_files_only=True), a.pool)
        json.dump(docs, open(a.docs, 'w'))
    done = {(r['context'], r['kv']) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    for run in a.runs.split(','):
        length, kv = int(run.split(':')[0]), run.split(':')[1]
        if (length, kv) in done: continue
        rc = subprocess.run([sys.executable, __file__, '--model', a.model, '--out', a.out, '--docs', a.docs,
                             '--child', run]).returncode
        if rc: print(f'FAIL {run} rc={rc}', flush=True)


if __name__ == '__main__':
    main()

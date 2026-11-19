#!/usr/bin/env python3
"""Verify one hardware recipe in TabbyAPI: the exact server config, torch memory capped to the card's budget, one
full-length chat request (image + documents + five invented-fact questions, see long_ctx.py) that must answer 5/5.

Runs inside the TabbyAPI container. The cap: torch.cuda.set_per_process_memory_fraction so torch can use at most
card GiB - headroom - 0.46 GiB non-torch overhead (measured on an idle RTX PRO 4500); exceeding it is a CUDA OOM.
  recipe_check.py --model DIR --card 24 --context 458752 --kv Q8 --out result.json
"""
import argparse, base64, json, os, re, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from long_ctx import FACTS, DEPTHS, grade

NON_TORCH_GIB = 0.46


def messages(tok, docs, length, image_b64):
    qs = '\n'.join(f'{i + 1}. {q}' for i, (_, q, _) in enumerate(FACTS))
    tail = ('\n\nAnswer the following questions using only the documents above. Give one short answer per line, '
            'numbered 1 to 5.\n' + qs)
    head = 'Below is a collection of documents (and an unrelated photo). Read them carefully; questions follow at the end.\n\n'
    from PIL import Image
    import io
    w, h = Image.open(io.BytesIO(base64.b64decode(image_b64))).size
    image_tokens = (w // 32) * (h // 32)                      # patch 16, merge 2 (resized to multiples of 32)
    budget = length - len(tok.encode(head + tail)) - image_tokens - 4096  # chat template, separators, answer (estimate drifts with many documents)
    chosen, used = [], 0
    for d in docs:
        n = len(tok.encode(d)) + 8
        if used + n > budget: break
        chosen.append(d); used += n
    body = [f'[Document {i + 1}]\n{d}' for i, d in enumerate(chosen)]
    for (fact, _, _), depth in zip(FACTS, DEPTHS):
        k = min(len(body) - 1, int(depth * len(body)))
        paras = body[k].split('\n\n'); paras.insert(max(1, len(paras) // 2), fact); body[k] = '\n\n'.join(paras)
    text = head + '\n\n'.join(body) + tail
    return [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image_b64}},
                                         {'type': 'text', 'text': text}]}], used


def main():
    p = argparse.ArgumentParser()
    for k in ('--model', '--out', '--kv', '--docs', '--image', '--tabby', '--config'): p.add_argument(k, required=k in ('--model', '--out'))
    p.add_argument('--card', type=float, required=True); p.add_argument('--context', type=int, default=0)
    p.add_argument('--card-mib', type=int, help='exact device total to cap against (nvidia-smi MiB); default card*1024')
    p.add_argument('--headroom', type=float, default=1.0); p.add_argument('--port', type=int, default=5600)
    a = p.parse_args()
    a.kv = a.kv or 'Q8'; a.tabby = a.tabby or '/tabby'
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    total_card = a.card_mib / 1024 if a.card_mib else a.card
    budget = total_card - a.headroom - NON_TORCH_GIB
    if a.config:   # a published recipe file, used as-is; only server/model location is added
        body = open(a.config).read()
        a.context = int(re.search(r'max_seq_len: (\d+)', body).group(1)); a.kv = re.search(r'cache_mode: (\w+)', body).group(1)
        body = re.sub(r'model_name: .*', 'model_name: ' + json.dumps(os.path.basename(a.model)), body)
        body = body.replace('model:\n', 'model:\n  model_dir: ' + json.dumps(os.path.dirname(a.model)) + '\n  backend: exllamav3\n', 1)
        open('/tmp/recipe.yml', 'w').write(f'network:\n  host: 127.0.0.1\n  port: {a.port}\n  disable_auth: true\n'
                                           '  disable_fetch_requests: true\n  web_ui: false\n' + body)
    cfg = open('/workspace/desktop/tabby/config.yml.tmpl').read()
    cfg = (cfg.replace('MODEL_PARENT', json.dumps(os.path.dirname(a.model))).replace('MODEL_NAME', json.dumps(os.path.basename(a.model)))
              .replace('PORT', str(a.port)).replace('MAX_SEQ_LEN', str(a.context))
              .replace('  chunk_size: 256', '  chunk_size: 2048').replace('  max_batch_size: 4', '  max_batch_size: 1'))
    cfg = cfg.replace('  tool_format: qwen3_coder', f'  tool_format: qwen3_coder\n  cache_mode: {a.kv}\n  vision: true')
    if not a.config: open('/tmp/recipe.yml', 'w').write(cfg)
    # sitecustomize applies the cap inside the Tabby process before any model memory is allocated
    os.makedirs('/tmp/cap', exist_ok=True)
    open('/tmp/cap/sitecustomize.py', 'w').write(
        'import atexit, os, torch\n'
        f'torch.cuda.set_per_process_memory_fraction({budget / total:.6f}, 0)\n'
        "atexit.register(lambda: open('/tmp/peak.txt','w').write(str(torch.cuda.max_memory_allocated())))\n")
    env = dict(os.environ, PYTHONPATH='/tmp/cap:' + os.environ.get('PYTHONPATH', ''))
    srv = subprocess.Popen([sys.executable, 'main.py', '--config', '/tmp/recipe.yml'], cwd=a.tabby, env=env,
                           stdout=open('/tmp/tabby.log', 'w'), stderr=subprocess.STDOUT)
    res = dict(model=os.path.basename(a.model), card_gib=a.card, context=a.context, kv=a.kv, headroom_gib=a.headroom,
               torch_budget_gib=round(budget, 2), status='fail')
    try:
        for _ in range(900):
            if srv.poll() is not None: raise RuntimeError('tabby exited during load: ' + open('/tmp/tabby.log').read()[-1500:])
            try: urllib.request.urlopen(f'http://127.0.0.1:{a.port}/v1/models', timeout=2).read(); break
            except Exception: time.sleep(2)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        msgs, doc_tokens = messages(tok, json.load(open(a.docs)), a.context, base64.b64encode(open(a.image, 'rb').read()).decode())
        body = dict(model=os.path.basename(a.model), messages=msgs, max_tokens=120, temperature=0,
                    chat_template_kwargs={'enable_thinking': False})
        t0 = time.time()
        r = urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',
                                   json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=7200)
        d = json.load(r); answer = d['choices'][0]['message']['content']
        ok, refusal = grade(answer)
        res.update(status='ok', seconds=round(time.time() - t0, 1), usage=d.get('usage'), answer=answer, correct=sum(ok),
                   per_fact=ok, refusal=refusal, doc_tokens=doc_tokens)
        # short generation, to see the decode speed of this config (MTP drafting on or off)
        speed = []
        for prompt in ('Write a Python function that parses an ISO 8601 date string without using datetime, with tests.',
                       'Explain in three paragraphs why the sky is blue.'):
            body = dict(model=os.path.basename(a.model), messages=[{'role': 'user', 'content': prompt}], max_tokens=400,
                        temperature=0, chat_template_kwargs={'enable_thinking': False})
            t0 = time.time()
            d = json.load(urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{a.port}/v1/chat/completions',
                          json.dumps(body).encode(), {'Content-Type': 'application/json'}), timeout=600))
            n = d['usage']['completion_tokens']; speed.append(dict(prompt=prompt[:40], tokens=n, tok_s=round(n / (time.time() - t0), 1)))
        res['speed'] = speed
    except Exception as e:
        res['error'] = repr(e)[-2000:]
    finally:
        srv.terminate()
        try: srv.wait(60)
        except Exception: srv.kill()
        if os.path.exists('/tmp/peak.txt'):
            peak = int(open('/tmp/peak.txt').read()); res['torch_peak_gib'] = round(peak / 2**30, 2)
            res['card_estimate_gib'] = round(peak / 2**30 + NON_TORCH_GIB, 2)
        res['tabby_log_tail'] = open('/tmp/tabby.log').read()[-1500:]
    res['pass'] = res['status'] == 'ok' and res.get('correct') == 5
    json.dump(res, open(a.out, 'w'), indent=1)
    print(f"RECIPE {res['model']} {a.card:g}GB {a.context // 1024}k {a.kv}: pass={res['pass']} correct={res.get('correct')} "
          f"card~{res.get('card_estimate_gib')} GiB budget {res['torch_budget_gib']} {res.get('seconds')}s speed={[x['tok_s'] for x in res.get('speed', [])]} err={res.get('error', '')[:200]}", flush=True)


if __name__ == '__main__':
    main()

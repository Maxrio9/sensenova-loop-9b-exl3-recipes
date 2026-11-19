#!/usr/bin/env python3
"""Peak GPU memory of the BF16 vision tower per image size (the preprocessor allows up to 16,777,216 pixels).
Each size runs in a fresh process: load the vision component only, embed one image, report peak PyTorch allocation
above the loaded tower and the number of image tokens."""
import argparse, json, subprocess, sys
import torch
torch.set_grad_enabled(False)


def child(model_dir, side, image):
    from PIL import Image
    from exllamav3 import Config, Model, Tokenizer
    cfg = Config.from_directory(model_dir)
    vision = Model.from_config(cfg, component='vision'); vision.load(progressbar=False)
    tok = Tokenizer.from_config(cfg)
    torch.cuda.synchronize(); loaded = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    img = Image.open(image).convert('RGB').resize((side, side), Image.BICUBIC)
    emb = vision.get_image_embeddings(tokenizer=tok, image=img)
    torch.cuda.synchronize()
    print('RESULT ' + json.dumps({'side': side, 'pixels': side * side, 'image_tokens': int(emb.embeddings.shape[0]),
                                  'tower_gib': loaded / 2**30, 'peak_above_tower_gib': (torch.cuda.max_memory_allocated() - loaded) / 2**30}), flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--image', required=True)
    p.add_argument('--out', required=True); p.add_argument('--child', type=int); a = p.parse_args()
    if a.child: child(a.model, a.child, a.image); return
    rows = []
    for side in (1024, 2048, 2896, 4096):
        out = subprocess.run([sys.executable, __file__, '--model', a.model, '--image', a.image, '--out', a.out, '--child', str(side)],
                             capture_output=True, text=True)
        r = [json.loads(l[7:]) for l in out.stdout.splitlines() if l.startswith('RESULT ')]
        rows += r or [{'side': side, 'error': (out.stderr or out.stdout)[-800:]}]
        print(rows[-1], flush=True)
    json.dump(rows, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

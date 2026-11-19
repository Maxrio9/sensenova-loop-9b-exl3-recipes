# SenseNova-Loop-9B on your GPU

Ready-made [TabbyAPI](https://github.com/theroyallab/tabbyAPI) configs for running [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B)
with as much context as your graphics card can hold. Text and images, quantized with EXL3 ([model files](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3)).

## Pick your card

| Your GPU | Model version | Context, GPU used only for the model | Context, GPU also runs your desktop |
|---|---|---:|---:|
| 12 GB | `4.0bpw` | 160k tokens ([config](configs/tabby-12gb.yml)) | 128k ([config](configs/tabby-12gb-desktop.yml)) |
| 16 GB | `5.0bpw` | 268k ([config](configs/tabby-16gb.yml)) | 232k ([config](configs/tabby-16gb-desktop.yml)) |
| 24 GB | `6.0bpw` | 504k ([config](configs/tabby-24gb.yml)) | 472k ([config](configs/tabby-24gb-desktop.yml)) |
| 32 GB | `8.0bpw` | 512k, the model's maximum ([config](configs/tabby-32gb.yml)) | 512k ([config](configs/tabby-32gb-desktop.yml)) |

**If your monitor is plugged into this GPU, use the desktop config.** It keeps 1 GB free for your screen and apps.
Want several chats at once (for example agents running in parallel)? Each config has a `-4x` variant next to it
(e.g. [tabby-24gb-4x.yml](configs/tabby-24gb-4x.yml)): four requests at a time, sharing slightly less total context.

All versions stay close to the original model; 6.0 and 8.0 are practically indistinguishable from it.

## Install

You need an NVIDIA GPU, Python 3.12 and the CUDA toolkit (the ExLlamaV3 build compiles CUDA code).

1. **Install TabbyAPI** and run it once, so it creates its environment:
   ```bash
   git clone https://github.com/theroyallab/tabbyAPI
   cd tabbyAPI
   ./start.sh        # Windows: start.bat. Stop it again once it has installed everything.
   ```
2. **Install ExLlamaV3 with SenseNova Loop support** into TabbyAPI's environment (this model is not in the official release yet; this takes a few minutes):
   ```bash
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install --no-deps --force-reinstall "git+https://github.com/Maxrio9/exllamav3.git@sensenova-loop"
   ```
3. **Download the model version for your card** (here the 24 GB one):
   ```bash
   hf download Maxrio/SenseNova-Loop-9B-exl3 --revision 6.0bpw --local-dir models/SenseNova-Loop-9B-exl3-6.0bpw
   ```
4. **Use the config:** copy `config_sample.yml` to `config.yml`, replace its `model:` section with the one from your config file,
   and add `model_dir: models` to it. Then start TabbyAPI again (`./start.sh`). It serves an OpenAI-compatible API on port 5000.

Tested with TabbyAPI at commit `4728158`. If you later update TabbyAPI with `--update-deps`, repeat step 2.

## Good to know

- **About 0.75 GB stays free** as a safety margin (plus 1 GB in the desktop configs). If other programs use the GPU heavily
  (games, video, a browser with many tabs), use the desktop config or lower `cache_size` and `max_seq_len`.
- **Images up to about 4 megapixels** (2048 x 2048) are covered. Larger images need about 0.1 GB extra per megapixel while they are read.
- **Thinking mode** is off unless you ask for it: send `"chat_template_kwargs": {"enable_thinking": true}` in the request.
  Tool calls work out of the box.

## Make your own config

`tools/recipes.py` prints a config for your exact setup (needs only Python):

```bash
python tools/recipes.py --card 24                 # 24 GB card used only for the model
python tools/recipes.py --card 24 --desktop       # ... that also drives your monitor
python tools/recipes.py --card 16 --no-vision     # text only: no vision tower, ~45k more tokens
```

| Flag | What it does |
|---|---|
| `--card 12/16/24/32` or `--card-mib N` | Your card; `N` is the total from `nvidia-smi --query-gpu=memory.total --format=csv` |
| `--desktop` | Keep 1 GB free for your display |
| `--no-vision` | Text only: skip the vision tower (about 1.4 GB more for context) |
| `--branch 8.0/6.0/5.0/4.0` | Trade quality for context: higher = closer to the original, lower = more context |
| `--kv q8/fp16` | KV cache type; Q8 (default) costs no measurable quality and holds 1.8x the context of FP16 |
| `--parallel N` | Serve N requests at once |
| `--image-mp N` | Largest image you will send, in megapixels (default 4) |

## Test results

- **Quality:** 8.0 bpw matches the original model within measurement noise; 6.0 and 5.0 bpw are closer to it than a community
  FP8 checkpoint; 4.0 bpw has about a quarter of the error of the official 4-bit AWQ release. Benchmarks (IFEval, MATH-500,
  German math, tool calling, extraction) are within noise of the original for all tested versions.
- **The configs work as stated:** the 12 and 16 GB desktop configs ran unchanged in TabbyAPI with memory capped to the card, a 4 MP image
  and a prompt filling the whole context: all five facts hidden in the text were found, inside the memory limit. The memory model behind all
  configs matched every real run within 0.4 GB.
- **Q8 KV cache is free:** quality with Q8 is identical to FP16 in our measurements; Q4 is measurably worse, so it is not used.
- **Long context:** five facts hidden at 10-90% depth were all found at 128k (FP16 and Q8 cache), 226k and 458k; at the full 512k, 4 of 5.

## More detail

How the numbers were measured and checked, the memory formula, quality measurements and all raw data:
[docs/DETAILS.md](docs/DETAILS.md).

Apache-2.0, like the model. Credit for SenseNova-Loop-9B goes to the SenseNova team.

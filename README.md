# SenseNova-Loop-9B on your GPU

Ready-made [TabbyAPI](https://github.com/theroyallab/tabbyAPI) configs for running [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B)
with as much context as your graphics card can hold. Text and images, quantized with EXL3 ([model files](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3)).

## Pick your card

| Your GPU | Model version | Context | Config |
|---|---|---:|---|
| 12 GB | `4.0bpw` | 128k tokens | [tabby-12gb.yml](configs/tabby-12gb.yml) |
| 16 GB | `5.0bpw` | 232k tokens | [tabby-16gb.yml](configs/tabby-16gb.yml) |
| 24 GB | `6.0bpw` | 472k tokens | [tabby-24gb.yml](configs/tabby-24gb.yml) |
| 32 GB | `8.0bpw` | 512k tokens (the model's maximum) | [tabby-32gb.yml](configs/tabby-32gb.yml) |

Want several chats at once (for example agents running in parallel)? Use the `-4x` config next to yours, e.g.
[tabby-24gb-4x.yml](configs/tabby-24gb-4x.yml): four requests at a time, sharing slightly less total context.

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

- **The configs leave about 1.75 GB of the card free**: 1 GB for your desktop and display, the rest as a safety margin.
  On a server without a display you can give the model more (see below). If other programs use the GPU heavily (games, video,
  a browser with many tabs), lower `cache_size` and `max_seq_len` in the config.
- **Images up to about 4 megapixels** (2048 x 2048) are covered. Much larger images need extra memory while they are read, about
  0.1 GB per megapixel; resize them, or lower `cache_size` a little.
- **Thinking mode** is off unless you ask for it: send `"chat_template_kwargs": {"enable_thinking": true}` in the request.
  Tool calls work out of the box (`tool_format: qwen3_coder` is set).
- **Different card size, or a headless server?** Compute your own config:
  ```bash
  python tools/recipes.py --card-mib 24564 --branch 6.0 --reserve 0.75
  ```
  (`--card-mib` is the total from `nvidia-smi --query-gpu=memory.total --format=csv`; `--reserve 0.75` drops the 1 GB display share.)

## More detail

How the numbers were measured and checked, the memory formula, quality measurements and all raw data:
[docs/DETAILS.md](docs/DETAILS.md).

Apache-2.0, like the model. Credit for SenseNova-Loop-9B goes to the SenseNova team.

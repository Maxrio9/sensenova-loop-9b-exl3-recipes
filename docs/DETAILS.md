# SenseNova-Loop-9B EXL3 recipes: details

Reference for agents, tool builders and anyone who wants to check or adapt the recipes: every number, how it was measured, and the raw data. For setup, see the [README](../README.md).

TabbyAPI configs that give [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) as much context as each card class can hold,
using the EXL3 quants at [Maxrio/SenseNova-Loop-9B-exl3](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3), with the memory measurements behind them.

## Pick your card

Desktop configs (`*-desktop*.yml`, the card also drives a display). The default configs for a GPU used only for the model
drop the 1.0 GiB display reserve and get about 31k more Q8 tokens: 136k / 232k / 456k / 512k (pool 630k).
All configs draft with the MTP head ([Speculative decoding](#speculative-decoding-mtp)), which costs 0.48 GiB plus 7% of the KV pool.

| Card | Branch | KV cache | Longest request | KV pool, 1 request | KV pool, 4 parallel | Config |
|---|---|---|---:|---:|---:|---|
| 12 GB | `4.0bpw` | Q8 | 104k | 106,496 | 90,112 | [`tabby-12gb-desktop.yml`](../configs/tabby-12gb-desktop.yml) / [`-4x`](../configs/tabby-12gb-desktop-4x.yml) |
| 16 GB | `5.0bpw` | Q8 | 204k | 208,896 | 192,512 | [`tabby-16gb-desktop.yml`](../configs/tabby-16gb-desktop.yml) / [`-4x`](../configs/tabby-16gb-desktop-4x.yml) |
| 24 GB | `6.0bpw` | Q8 | 424k | 434,176 | 417,792 | [`tabby-24gb-desktop.yml`](../configs/tabby-24gb-desktop.yml) / [`-4x`](../configs/tabby-24gb-desktop-4x.yml) |
| 32 GB | `8.0bpw` | Q8 | 512k (native max) | 602,112 | 585,728 | [`tabby-32gb-desktop.yml`](../configs/tabby-32gb-desktop.yml) / [`-4x`](../configs/tabby-32gb-desktop-4x.yml) |

The KV pool (`cache_size`) is shared by all requests running at the same time; one request can use up to `max_seq_len` of it.
The `-4x` configs serve 4 requests at once: each parallel slot costs 0.167 GiB, about 5k tokens of Q8 pool.
On 32 GB the pool is larger than the model's 512k window, so it can hold one full-length conversation and still serve others.

Why these branches: 8.0 bpw is indistinguishable from BF16, 6.0 and 5.0 bpw are closer to BF16 than an FP8 checkpoint, and 4.0 bpw
still has a quarter of the official 4-bit AWQ's error ([quality numbers](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3)).
Q8 KV cache costs no measurable quality; Q4 KV does (see [Quality](#quality)).

## What each recipe leaves free

The desktop configs leave **1.74 GiB** of the card unused, the default configs **0.74 GiB** (no display share):

| Reserve | GiB | Why |
|---|---:|---|
| Desktop / display (desktop configs only) | 1.0 | A Windows or Linux desktop driving the screen; `--desktop` in tools/recipes.py |
| Allocator and library slack | 0.3 | PyTorch fragmentation, kernel workspaces |
| One image up to 4 MP | 0.44 | Measured vision tower working memory while encoding a 4 MP image (about 0.11 GiB per megapixel) |

The model allows images up to 16.8 MP (4096 x 4096), which needs 1.8 GiB while encoding. For larger images either lower `cache_size`
(0.106 GiB per MP above 4 MP, about 3.5k Q8 tokens) or cap the image size in the model's `preprocessor_config.json` (`size.longest_edge`
is the maximum pixel count; 4,194,304 = 2048 x 2048). TabbyAPI has no image size limit of its own. Other programs using the GPU,
for example a browser with hardware acceleration, also come out of the reserve.

To compute your own recipe (exact card size from `nvidia-smi --query-gpu=memory.total --format=csv`, any branch, KV type, reserve, parallel slots):

```bash
python tools/recipes.py --card-mib 16376 --branch 6.0 --kv q8 --desktop --parallel 2
```

## How to run

See the [README](../README.md#install).

## Memory model

For one request, vision tower loaded, delta-net state in FP32, prefill chunk 2048:

```
GPU memory (GiB) = base[branch] + per_token[KV] x cache_size + 0.2 + 0.167 x (parallel - 1)
                 + MTP drafting: 0.23 + 0.25 + per_token[KV] / 14 x cache_size
```

MTP terms: the 4-bit head and its buffers (0.23 GiB measured), the sparse draft head at its full 16,384 columns (fp16 unembedding
columns and embedding rows, 16 KiB per column), and the head's own KV cache: one attention layer against the target's 14 attention
visits (8 full-attention layers, 6 of them inside the looped block).

| KV cache | GiB per 1k tokens |
|---|---:|
| FP16 | 0.0547 |
| Q8 | 0.0303 |
| Q6 | 0.0235 |
| Q4 | 0.0167 |

| Branch | base, Q8 KV (GiB) |
|---|---:|
| 8.0bpw | 10.27 |
| 6.0bpw | 7.70 |
| 5.0bpw | 6.90 |
| 4.0bpw | 6.08 |

`base` covers the weights, the 0.9 GiB vision tower, buffers, and 470 MiB of non-PyTorch memory (CUDA context and kernels).

How it was measured:
- PyTorch peak allocation for every branch x KV type (FP16/Q8/Q6/Q4) x context (8k/32k/64k/128k) on an NVIDIA DGX Spark (GB10), with one real request and one image each ([data/vram-sweeps](../data/vram-sweeps)). 5.0 bpw was measured at one point and predicted elsewhere.
- The non-PyTorch part with nvidia-smi on an otherwise idle RTX PRO 4500 Blackwell: 455-473 MiB at every context from 1k to 128k. PyTorch peaks on the Spark match the RTX PRO 4500 within 0.01 GiB.
- The formula reproduces every measured sweep point within 0.06 GiB. In full TabbyAPI runs up to 523k tokens the measured memory was at most 0.06 GiB above and up to 0.4 GiB below the prediction.
- Parallel requests: same pool, 1/4/8 requests, 0.167 GiB per extra request at both steps ([data/batchmem-5.0.json](../data/batchmem-5.0.json)).
- Image encoding: 1, 4, 8.4 and 16.8 MP images, linear at 0.106 GiB per MP ([data/vision-mem.json](../data/vision-mem.json)).

## Verification

| Card | What ran | Result |
|---|---|---|
| 12 GB | [`tabby-12gb-desktop.yml`](../configs/tabby-12gb-desktop.yml) without MTP (earlier version): 130,343-token prompt + 4 MP image | 5/5 facts, 10.2 GiB (limit with reserves 10.7) |
| 16 GB | [`tabby-16gb-desktop.yml`](../configs/tabby-16gb-desktop.yml) without MTP (earlier version): 237,385-token prompt + 4 MP image | 5/5 facts, 14.1 GiB (limit 14.7) |
| 16 GB, MTP | [`tabby-16gb-desktop.yml`](../configs/tabby-16gb-desktop.yml) as published, with MTP drafting: 204,954-token prompt + 4 MP image | 5/5 facts, 14.0 GiB (limit 14.2, predicted 14.2); right after start 42 tok/s on code, 29 on prose ([data](../data/recipe-checks/16gb-5.0-204k-mtp.json)) |
| 24 GB | Predicted 22.2 GiB. Closest run: 6.0bpw at 448k in TabbyAPI, small image | 5/5 facts, 21.2 GiB measured vs 21.5 predicted |
| 32 GB | Predicted 30.0 GiB. Closest run: 8.0bpw at the full 512k in TabbyAPI, small image | 25.6 GiB measured vs 26.0 predicted; 4 of 5 facts |

The checks ran in TabbyAPI with PyTorch memory capped to the card's total minus 1.0 GiB (display), 0.3 GiB (slack) and 0.46 GiB
(non-PyTorch memory), with one request that fills `max_seq_len`: the model's own long answers as documents, five ordinary facts about
invented places and people at 10/30/50/70/90% depth, an image, and five questions (thinking off, greedy). A run passes if the server
stays inside the cap and answers all five. The facts and questions avoid codes or "secrets", which this model tends to refuse.
This is a sanity check of memory and retrieval, not a long-context benchmark. FP16 and Q8 KV also found 5/5 at 4k and 128k.
All runs were on a DGX Spark; the card totals are typical `nvidia-smi` values (12,282 / 16,376 / 24,564 / 32,607 MiB), not tests on those cards.

## Speculative decoding (MTP)

The model ships a multi-token-prediction head: one extra transformer layer that guesses the next tokens from the model's last hidden
state. Each branch has it as `model-mtp.safetensors`, quantized to 4 bits (0.12 GB; 0.49 GB in BF16). The configs draft with it:

```yaml
draft_model:
  draft_mode: mtp
  draft_num_tokens: 4        # tokens guessed per round
  draft_cache_mode: Q8
  hybrid_draft_tokens: 7     # when the model repeats text from its context, propose the next 7 tokens of it
  sparse_draft_head: 16384   # draft over the 16,384 most frequent tokens; verification always uses the full vocabulary
```

The model checks every drafted token in one pass, so the output is the same as without drafting. In our tests greedy output was
token-identical to plain decoding on all ten tasks for every setting below (ExLlamaV3 fork, `EXL3_INT8_GEMV=0`, FP32 delta-net
state). With ExLlamaV3's default INT8 decode path, rare near-ties between two tokens can resolve differently with and without drafting;
both results are valid outputs of the model, so this matters only if you need bit-identical reruns.

Speed, 5.0bpw on a DGX Spark (GB10), greedy, 320 new tokens per task, plain decoding 21-22 tok/s
([data/mtp](../data/mtp/spec-5.0-sparse-sweep.txt)):

| Drafting | Extra GiB | Mean speedup | Code | Prose | Chat | Math (thinking) | Edit 60 lines | Quote a document |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MTP, full vocabulary | 0.23 | 1.68x | 2.03x | 1.03x | 1.02x | 1.88x | 2.07x | 1.85x |
| MTP, sparse 8,192 | 0.36 | 2.31x | 2.88x | 1.46x | 1.55x | 2.69x | 2.74x | 2.43x |
| **MTP, sparse 16,384 (configs)** | **0.48** | **2.38x** | **2.85x** | **1.57x** | **1.51x** | **2.76x** | **2.75x** | **2.48x** |
| MTP, sparse 32,768 | 0.74 | 2.42x | 2.83x | 1.56x | 1.46x | 2.70x | 2.68x | 2.45x |
| MTP, sparse 65,536 | 1.25 | 2.50x | 2.87x | 1.55x | 1.51x | 2.69x | 3.25x | 2.62x |

All rows: 4 draft tokens, hybrid 7. Most of the gain over the plain MTP head comes from the sparse head: drafting through the full
248,320-token output layer costs nearly as much as the guess saves. 4 draft tokens with hybrid 7 was the fastest setting in repeated
6.0bpw runs (2.4-2.5x); 3 draft tokens accept more (about 65% vs 52% on code) but gain less per round.
The sparse head picks its tokens from counts it keeps while generating; they are saved under `~/.cache/exllamav3/token_stats`.
Until it has seen a few thousand tokens, drafting runs at about 1.7x.

Measured alternatives on 6.0bpw (same tasks): n-gram drafting alone 1.3x (only edits and quotes gain), Qwen3.5-0.8B as a separate
draft model 1.64x for 1.6 GiB, Qwen3.5-2B 1.21x for 3.7 GiB. The MTP head is faster and smaller than both.

## Quality

Held-out KL divergence from BF16 (240 conversations generated by the model, 152,145 scored tokens; lower is better):

| | KL vs BF16 | Top-1 agreement |
|---|---:|---:|
| 8.0bpw | 0.00016 | 99.7% |
| 6.0bpw | 0.0015 | 99.0% |
| 6.0bpw, Q8 KV cache | 0.0015 | 99.0% |
| 6.0bpw, Q4 KV cache | 0.0040 | 98.4% |
| 5.0bpw | 0.0030 | 98.6% |
| 4.0bpw | 0.0114 | 97.3% |
| BF16 run twice (noise floor) | 0.00032 | 99.6% |

KV-cache rows from [data/kv-quality](../data/kv-quality); conversations are up to about 10k tokens long.

## Not covered yet

- MTP speed on consumer cards: measured on GB10 only. The draft head's share of a round is smaller on cards with faster memory, so
  the speedup there can differ.
- Tests on actual 12/16/24 GB consumer cards, and Windows. The display reserve is an estimate (Windows desktops commonly use 0.5-1.7 GB).

## Tools

- [`tools/recipes.py`](../tools/recipes.py): the memory model and config generator.
- [`tools/recipe_check.py`](../tools/recipe_check.py): runs a config in TabbyAPI under a memory cap with a full-context request.
- [`tools/vram.py`](../tools/vram.py), [`tools/batch_mem.py`](../tools/batch_mem.py), [`tools/vision_mem.py`](../tools/vision_mem.py), [`tools/long_ctx.py`](../tools/long_ctx.py): the measurements.

The tools expect the ExLlamaV3 fork and, for the checks, a TabbyAPI checkout; paths inside are those of our test containers.

## License

Apache-2.0, like the model. Credit for SenseNova-Loop-9B goes to the SenseNova team.

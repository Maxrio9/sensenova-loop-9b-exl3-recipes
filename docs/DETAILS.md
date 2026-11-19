# SenseNova-Loop-9B EXL3 recipes: details

Reference for agents, tool builders and anyone who wants to check or adapt the recipes: every number, how it was measured, and the raw data. For setup, see the [README](../README.md).

TabbyAPI configs that give [SenseNova-Loop-9B](https://huggingface.co/sensenova/SenseNova-Loop-9B) as much context as each card class can hold,
using the EXL3 quants at [Maxrio/SenseNova-Loop-9B-exl3](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3), with the memory measurements behind them.

## Pick your card

| Card | Branch | KV cache | Longest request | KV pool, 1 request | KV pool, 4 parallel | Config |
|---|---|---|---:|---:|---:|---|
| 12 GB | `4.0bpw` | Q8 | 128k | 131,072 | 114,688 | [`tabby-12gb.yml`](../configs/tabby-12gb.yml) / [`-4x`](../configs/tabby-12gb-4x.yml) |
| 16 GB | `5.0bpw` | Q8 | 232k | 237,568 | 221,184 | [`tabby-16gb.yml`](../configs/tabby-16gb.yml) / [`-4x`](../configs/tabby-16gb-4x.yml) |
| 24 GB | `6.0bpw` | Q8 | 472k | 483,328 | 466,944 | [`tabby-24gb.yml`](../configs/tabby-24gb.yml) / [`-4x`](../configs/tabby-24gb-4x.yml) |
| 32 GB | `8.0bpw` | Q8 | 512k (native max) | 659,456 | 643,072 | [`tabby-32gb.yml`](../configs/tabby-32gb.yml) / [`-4x`](../configs/tabby-32gb-4x.yml) |

The KV pool (`cache_size`) is shared by all requests running at the same time; one request can use up to `max_seq_len` of it.
The `-4x` configs serve 4 requests at once: each parallel slot costs 0.167 GiB, about 5.5k tokens of Q8 pool.
On 32 GB the pool is larger than the model's 512k window, so it can hold one full-length conversation and still serve others.

Why these branches: 8.0 bpw is indistinguishable from BF16, 6.0 and 5.0 bpw are closer to BF16 than an FP8 checkpoint, and 4.0 bpw
still has a quarter of the official 4-bit AWQ's error ([quality numbers](https://huggingface.co/Maxrio/SenseNova-Loop-9B-exl3)).
Q8 KV cache costs no measurable quality; Q4 KV does (see [Quality](#quality)).

## What each recipe leaves free

Every config leaves **1.75 GiB** of the card unused:

| Reserve | GiB | Why |
|---|---:|---|
| Desktop / display | 1.0 | A Windows or Linux desktop driving the screen; a headless server can give this to the pool (+33k Q8 tokens) |
| Allocator and library slack | 0.3 | PyTorch fragmentation, kernel workspaces |
| One image up to 4 MP | 0.45 | The vision tower needs 0.106 GiB of working memory per megapixel while it encodes an image |

The model allows images up to 16.8 MP (4096 x 4096), which needs 1.8 GiB while encoding. For larger images either lower `cache_size`
(0.106 GiB per MP above 4 MP, about 3.5k Q8 tokens) or cap the image size in the model's `preprocessor_config.json` (`size.longest_edge`
is the maximum pixel count; 4,194,304 = 2048 x 2048). TabbyAPI has no image size limit of its own. Other programs using the GPU,
for example a browser with hardware acceleration, also come out of the reserve.

To compute your own recipe (exact card size from `nvidia-smi --query-gpu=memory.total --format=csv`, any branch, KV type, reserve, parallel slots):

```bash
python tools/recipes.py --card-mib 16376 --branch 6.0 --kv q8 --reserve 1.75 --parallel 2
```

## How to run

See the [README](../README.md#install).

## Memory model

For one request, vision tower loaded, delta-net state in FP32, prefill chunk 2048:

```
GPU memory (GiB) = base[branch] + per_token[KV] x cache_size + 0.2 + 0.167 x (parallel - 1)
```

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
| 12 GB | [`tabby-12gb.yml`](../configs/tabby-12gb.yml) as published: 130,343-token prompt + 4 MP image | 5/5 facts, 10.2 GiB (limit with reserves 10.7) |
| 16 GB | [`tabby-16gb.yml`](../configs/tabby-16gb.yml) as published: 237,385-token prompt + 4 MP image | 5/5 facts, 14.1 GiB (limit 14.7) |
| 24 GB | Predicted 22.2 GiB. Closest run: 6.0bpw at 448k in TabbyAPI, small image | 5/5 facts, 21.2 GiB measured vs 21.5 predicted |
| 32 GB | Predicted 30.0 GiB. Closest run: 8.0bpw at the full 512k in TabbyAPI, small image | 25.6 GiB measured vs 26.0 predicted; 4 of 5 facts |

The checks ran in TabbyAPI with PyTorch memory capped to the card's total minus 1.0 GiB (display), 0.3 GiB (slack) and 0.46 GiB
(non-PyTorch memory), with one request that fills `max_seq_len`: the model's own long answers as documents, five ordinary facts about
invented places and people at 10/30/50/70/90% depth, an image, and five questions (thinking off, greedy). A run passes if the server
stays inside the cap and answers all five. The facts and questions avoid codes or "secrets", which this model tends to refuse.
This is a sanity check of memory and retrieval, not a long-context benchmark. FP16 and Q8 KV also found 5/5 at 4k and 128k.
All runs were on a DGX Spark; the card totals are typical `nvidia-smi` values (12,282 / 16,376 / 24,564 / 32,607 MiB), not tests on those cards.

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

- Speculative decoding with the model's MTP head: work in progress in the ExLlamaV3 fork. It will add about 0.5 GiB and change these numbers.
- Tests on actual 12/16/24 GB consumer cards, and Windows. The display reserve is an estimate (Windows desktops commonly use 0.5-1.7 GB).

## Tools

- [`tools/recipes.py`](../tools/recipes.py): the memory model and config generator.
- [`tools/recipe_check.py`](../tools/recipe_check.py): runs a config in TabbyAPI under a memory cap with a full-context request.
- [`tools/vram.py`](../tools/vram.py), [`tools/batch_mem.py`](../tools/batch_mem.py), [`tools/vision_mem.py`](../tools/vision_mem.py), [`tools/long_ctx.py`](../tools/long_ctx.py): the measurements.

The tools expect the ExLlamaV3 fork and, for the checks, a TabbyAPI checkout; paths inside are those of our test containers.

## License

Apache-2.0, like the model. Credit for SenseNova-Loop-9B goes to the SenseNova team.

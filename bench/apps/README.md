# V100 application benchmarks: DINOv3-H LoRA fine-tuning & SD3.5-Medium generation

Two real workloads, each run twice with **identical hyper-parameters** — once with
PyTorch SDPA, once with this repository's `flash-attn` (official package name,
sm_70 build) — to measure the actual before/after difference on a Tesla V100S-PCIE-32GB.

All code and data live on the AutoDL server under `/root/autodl-tmp`:

| Path | Content |
| --- | --- |
| `/root/autodl-tmp/code` | the four scripts, result JSONs, logs, generated images |
| `/root/autodl-tmp/models/dinov3-h` | DINOv3 ViT-H/16+ weights (ModelScope) |
| `/root/autodl-tmp/models/sd35m` | SD3.5-Medium pipeline components (ModelScope) |
| `/root/autodl-tmp/models/tiny-imagenet` | TinyImageNet-200 (HF mirror) |

## Environment

* Tesla V100S-PCIE-32GB (sm_70), driver 580.95.05
* PyTorch **2.8.0+cu128**, CUDA runtime 12.8, Python 3.12
* `transformers` 5.20.1, `diffusers` 0.41.0, `peft` 0.21.2, `datasets` 5.1.0, `modelscope` 1.40.2
* `flash_attn` **2.8.3** (this repository, imported as `flash_attn` / `flash_attn_2_cuda`)

## Models and datasets (domestic mirrors)

| Item | Source | Notes |
| --- | --- | --- |
| DINOv3-H/16+ | ModelScope `facebook/dinov3-vith16plus-pretrain-lvd1689m` | 3.36 GB `model.safetensors` |
| SD3.5-Medium | ModelScope `AI-ModelScope/stable-diffusion-3.5-medium` | fp16 transformer + T5-XXL/CLIP encoders + VAE |
| TinyImageNet-200 | HF mirror `https://hf-mirror.com`, dataset `zh-plus/tiny-imagenet` | 100k train / 10k valid, 64×64, 200 classes |

Commands used to populate the caches:

```bash
# DINOv3-H + SD3.5-Medium (domestic ModelScope, no proxy needed)
/root/autodl-tmp/envs/task/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download("facebook/dinov3-vith16plus-pretrain-lvd1689m",
                  cache_dir="/root/autodl-tmp/models/dinov3-h")
snapshot_download("AI-ModelScope/stable-diffusion-3.5-medium",
                  cache_dir="/root/autodl-tmp/models/sd35m",
                  allow_patterns=["transformer/*", "vae/*", "text_encoder*/*",
                                  "tokenizer*/*", "scheduler/*", "model_index.json"])
PY

# TinyImageNet via HF mirror
HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/models/hf \
/root/autodl-tmp/envs/task/bin/python - <<'PY'
from datasets import load_dataset
load_dataset("zh-plus/tiny-imagenet").save_to_disk("/root/autodl-tmp/models/tiny-imagenet")
PY
```

## Scripts

| File | Attention backend |
| --- | --- |
| `dinov3-finetuning.py` | `transformers` SDPA |
| `dinov3-finetuning-flash.py` | `flash_attn_func` (patches `DINOv3ViTAttention.forward`, RoPE preserved) |
| `sd35m-gen.py` | `diffusers` native SDPA |
| `sd35m-gen-flash.py` | `torch.nn.functional.scaled_dot_product_attention` routed to `flash_attn_func` when shapes are supported (head_dim ∈ {16,32,64,128,256}, fp16, no mask); otherwise SDPA |

### DINOv3-H LoRA (rank 64) on TinyImageNet

* Backbone: `DINOv3ViTModel` (ViT-H/16+, 224×224, 32 layers, 20 heads, hidden 1280, 4 registers)
* LoRA `r=64`, `alpha=128`, dropout 0, targets `q_proj/k_proj/v_proj/o_proj/up_proj/down_proj`; classifier head 1280→200
* Trainable parameters: **47.44 M**
* Batch 16, 5 warm-up + 30 measured optimizer steps, AdamW lr 1e-3, fp32 weights + AMP fp16 (fp16 weights alone produced NaNs in DINOv3 on V100), train subset 4096, val subset 1024

```bash
cd /root/autodl-tmp/code
/root/autodl-tmp/envs/task/bin/python dinov3-finetuning.py \
    --batch 16 --steps 30 --warmup 5 --train-size 4096 --val-size 1024
/root/autodl-tmp/envs/task/bin/python dinov3-finetuning-flash.py \
    --batch 16 --steps 30 --warmup 5 --train-size 4096 --val-size 1024
```

| Metric | SDPA | FlashAttention | Δ |
| --- | ---: | ---: | ---: |
| ms / training step | 393.10 | 393.72 | +0.16 % |
| samples / s | 40.70 | 40.64 | −0.15 % |
| peak GPU memory | 13 198.7 MiB | 13 220.1 MiB | +21.4 MiB |
| loss (first → last) | 5.263 → 3.830 | 5.263 → 3.807 | ≈ same |
| val accuracy (1024 imgs) | 73.44 % | 70.61 % | noise, 30 steps |

**Result: at 224×224 (201 tokens) the attention kernel is a small fraction of ViT-H
compute; this sm_70 FlashAttention build is performance-neutral for DINOv3 LoRA
fine-tuning (within 0.2 %), with identical memory.**

### SD3.5-Medium text-to-image

* 512×512, 20 inference steps, guidance 4.5, seed 1234, fp16
* 1 warm-up image + 3 measured images; median reported

```bash
cd /root/autodl-tmp/code
/root/autodl-tmp/envs/task/bin/python sd35m-gen.py       --steps 20 --iters 3 --warmup 1
/root/autodl-tmp/envs/task/bin/python sd35m-gen-flash.py --steps 20 --iters 3 --warmup 1
```

| Metric | SDPA | FlashAttention | Δ |
| --- | ---: | ---: | ---: |
| ms / 512×512 image | 2 764.6 | 2 994.4 | **+8.31 %** |
| images / s | 0.3617 | 0.3340 | −7.7 % |
| peak GPU memory | 18 106.3 MiB | 18 106.3 MiB | 0 |

Output quality: PSNR between the two same-seed images is **42.46 dB** (MSE 3.69,
max channel diff 66), i.e. visually equivalent; differences come from fp16
non-determinism between the two attention kernels.

**Result: for SD3.5-Medium generation on V100 the FlashAttention path is ~8 %
slower than PyTorch SDPA, with no memory change.** The attention workload
(short joint sequences, single image) does not amortize the kernel's overhead;
the pipeline is dominated by the MMDiT/VAE layers and text encoders.

## Analysis / known limitations

* Both tasks are **inference/short-sequence** workloads. The repo's earlier
  micro-benchmarks (`bench/bench_pareto.py`) show the kernel's advantage at large
  batch/sequence and in training backward; these app workloads do not hit that regime.
* DINOv3-H in pure fp16 produced NaN logits on V100; the benchmark uses fp32
  weights + AMP fp16 for both backends, so the comparison is fair but not the
  fastest possible configuration.
* `diffusers` 0.41 probes `flash_attn.flash_attn_interface._wrapped_flash_attn_forward/backward`
  for its built-in "flash" backend and falls back to SDPA when they are absent.
  The four scripts do **not** depend on that private API: the flash variants patch
  the SDPA/attention call sites directly. This is noted as a drop-in-compatibility gap.
* TinyImageNet images are 64×64 and are resized to 224×224 by the DINOv3 image
  processor; the benchmark measures compute, not accuracy convergence (30 steps).
* Multi-version wheels (CUDA 12.8/torch 2.8.0, CUDA 12.6/torch 2.6.0, CUDA
  12.4/torch 2.5.1) are published on the `v2.8.3` release; see the top-level
  `README.md` compatibility matrix. The request's `torch 1.6.0` is treated as a
  typo for **2.6.0** (torch 1.6 cannot run with CUDA 12.x / Python 3.12).

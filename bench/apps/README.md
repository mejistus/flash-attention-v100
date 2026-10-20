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
* The measurements below were rerun with the **API-aligned wheel** (`v2.8.3-v100-api`) in
  `/root/autodl-tmp/envs/api28`; the earlier custom-interface numbers are kept in
  `results/*.json` as a cross-check.

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
/root/autodl-tmp/envs/api28/bin/python dinov3-finetuning.py \
    --batch 16 --steps 30 --warmup 5 --train-size 4096 --val-size 1024
/root/autodl-tmp/envs/api28/bin/python dinov3-finetuning-flash.py \
    --batch 16 --steps 30 --warmup 5 --train-size 4096 --val-size 1024
```

| Metric | SDPA | FlashAttention | Δ |
| --- | ---: | ---: | ---: |
| ms / training step (mean of 2 runs) | 386.75 | 394.60 | +2.03 % |
| samples / s | 41.37 | 40.55 | −1.99 % |
| peak GPU memory | 13 198.7 MiB | 13 220.1 MiB | +21.4 MiB |
| loss (first → last) | 5.263 → 3.788–3.853 | 5.263 → 3.789–3.842 | ≈ same |
| val accuracy (1024 imgs) | 71.6–73.3 % | 69.8–71.6 % | noise, 30 steps |

Run-to-run values: SDPA 386.40 / 387.11 ms, flash 394.70 / 394.50 ms.  The earlier
custom-interface run measured 393.10 vs 393.72 ms (+0.16 %).  The flash absolute
time is stable (~394.6 ms in both runs); the SDPA baseline shifted with the
environment/GPU clocks, so read the training delta as **~0–2 %** — no meaningful
speedup, identical memory.

### SD3.5-Medium text-to-image

* 512×512, 20 inference steps, guidance 4.5, seed 1234, fp16
* 1 warm-up image + 3 measured images; median reported

```bash
cd /root/autodl-tmp/code
/root/autodl-tmp/envs/api28/bin/python sd35m-gen.py       --steps 20 --iters 3 --warmup 1
/root/autodl-tmp/envs/api28/bin/python sd35m-gen-flash.py --steps 20 --iters 3 --warmup 1
```

| Metric | SDPA | FlashAttention | Δ |
| --- | ---: | ---: | ---: |
| ms / 512×512 image | 2 768.6 | 3 005.2 | **+8.55 %** |
| images / s | 0.3612 | 0.3328 | −7.9 % |
| peak GPU memory | 18 106.3 MiB | 18 106.3 MiB | 0 |

Output quality: PSNR between the two same-seed images is **42.46 dB** (MSE 3.69,
max channel diff 66), i.e. visually equivalent; differences come from fp16
non-determinism between the two attention kernels.

**Result: for SD3.5-Medium generation on V100 the FlashAttention path is ~8.5 %
slower than PyTorch SDPA, with no memory change** (previous custom-interface run:
+8.31 %; the API-aligned package adds ~0.36 % to the flash path).  The attention
workload (short joint sequences, single image) does not amortize the kernel's
overhead; the pipeline is dominated by the MMDiT/VAE layers and text encoders.

## Inference vs LoRA training (summary)

Both rows use the same V100S-32GB and torch 2.8.0+cu128 environment (API-aligned
wheel).  Absolute milliseconds are not comparable between the two different
workloads; the meaningful number is the in-row FlashAttention vs SDPA delta.

| workload | mode | SDPA | FlashAttention | Δ |
| --- | --- | ---: | ---: | ---: |
| DINOv3-H LoRA(64), 224 px, batch 16 | training (fwd+bwd+AdamW) | 386.8 ms/step, 41.37 samples/s, 13.2 GiB | 394.6 ms/step, 40.55 samples/s, 13.2 GiB | +2.0 % |
| SD3.5-Medium, 512×512, 20 steps | inference | 2 768.6 ms/image, 0.3612 images/s, 17.7 GiB | 3 005.2 ms/image, 0.3328 images/s, 17.7 GiB | +8.6 % |

**Conclusion:** on V100, FlashAttention is not a speed win for either workload:
LoRA training is ~neutral (0–2 % depending on run/environment) and SD3.5 inference
is consistently ~8.5 % slower; peak memory is unchanged.  The kernel's advantage
remains large-batch/long-sequence and packed-varlen training memory (top-level
README).

## Analysis / known limitations

* DINOv3 training uses short 201-token sequences; SD3.5 is single-image inference.
  The repo's earlier micro-benchmarks (`bench/bench_pareto.py`) show the kernel's
  advantage at large batch/sequence and in training backward; these app workloads
  do not hit that regime.
* DINOv3-H in pure fp16 produced NaN logits on V100; the benchmark uses fp32
  weights + AMP fp16 for both backends, so the comparison is fair but not the
  fastest possible configuration.
* With the API-aligned wheel the official `_wrapped_flash_attn_forward/backward`
  custom ops exist, so `diffusers` 0.41 detects flash-attn instead of falling back
  to SDPA.  The four scripts still route SDPA/attention directly for an
  apples-to-apples benchmark.
* TinyImageNet images are 64×64 and are resized to 224×224 by the DINOv3 image
  processor; the benchmark measures compute, not accuracy convergence (30 steps).
* Multi-version wheels (CUDA 12.8/torch 2.8.0, CUDA 12.6/torch 2.6.0, CUDA
  12.4/torch 2.5.1) are published on the `v2.8.3-v100-api` release; see the
  top-level `README.md` compatibility matrix. The request's `torch 1.6.0` is
  treated as a typo for **2.6.0** (torch 1.6 cannot run with CUDA 12.x / Python 3.12).
* API-alignment overhead: flash path 393.72→394.60 ms/step (training, +0.22 %)
  and 2 994.4→3 005.2 ms/image (inference, +0.36 %); the SDPA script does not use
  flash-attn, so its absolute shift between envs is environment/clock variance.

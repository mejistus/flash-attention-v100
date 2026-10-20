#!/usr/bin/env python
"""SD3.5-Medium text-to-image generation benchmark (V100, fp16).

  sd35m-gen.py        -> diffusers' native attention (PyTorch SDPA)
  sd35m-gen-flash.py  -> the same pipeline with torch SDPA routed to this repo's
                         flash-attn (sm_70 drop-in) wherever the shapes allow.

Model cache (domestic mirror, ModelScope):
  /root/autodl-tmp/models/sd35m/...  (AI-ModelScope/stable-diffusion-3.5-medium)
"""
import argparse, json, os, time
import torch

USE_FLASH = False
SNAP = "/root/autodl-tmp/models/sd35m/models/AI-ModelScope--stable-diffusion-3.5-medium/snapshots/master"

def patch_sdpa_to_flash():
    import torch.nn.functional as F
    from flash_attn import flash_attn_func
    orig = F.scaled_dot_product_attention

    def flash_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                   scale=None, enable_gqa=False, **kw):
        if (attn_mask is None and not enable_gqa and query.is_cuda and query.dim() == 4
                and query.dtype == torch.float16 and query.shape[-1] % 8 == 0
                and query.shape == key.shape == value.shape):
            q = query.transpose(1, 2).contiguous()
            k = key.transpose(1, 2).contiguous()
            v = value.transpose(1, 2).contiguous()
            out = flash_attn_func(q, k, v, dropout_p=dropout_p,
                                  softmax_scale=scale, causal=is_causal)
            return out.transpose(1, 2)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kw)

    F.scaled_dot_product_attention = flash_sdpa
    print("[flash] routed torch SDPA -> flash_attn_func (shape-compatible calls)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a photograph of a red fox sitting in a snowy forest, highly detailed")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from diffusers import StableDiffusion3Pipeline
    if USE_FLASH:
        patch_sdpa_to_flash()
    pipe = StableDiffusion3Pipeline.from_pretrained(SNAP, torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")

    def gen_once():
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        return pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=args.guidance,
                    height=args.height, width=args.width, generator=g).images[0]

    for _ in range(args.warmup):
        gen_once()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    times = []
    img = None
    for _ in range(args.iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); img = gen_once(); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated() / 2**20
    import statistics
    mode = "flash" if USE_FLASH else "sdpa"
    img_path = f"/root/autodl-tmp/code/sd35m_{mode}.png"
    img.save(img_path)
    res = dict(mode=mode, steps=args.steps, size=[args.height, args.width],
               seed=args.seed, ms_per_image=statistics.median(times),
               images_per_s=1000.0 / statistics.median(times), peak_mib=peak,
               warmup=args.warmup, iters=args.iters, image=img_path,
               torch=torch.__version__, device=torch.cuda.get_device_name(0))
    print(json.dumps(res, indent=1))
    out = args.out or f"/root/autodl-tmp/code/results_sd35m_{mode}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main()

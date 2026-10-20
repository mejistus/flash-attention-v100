#!/usr/bin/env python
"""DINOv3-H ViT (LoRA rank 64) fine-tuning on TinyImageNet-200.

Two variants share every hyper-parameter; only the attention backend differs:
  dinov3-finetuning.py        -> transformers SDPA (native PyTorch)
  dinov3-finetuning-flash.py  -> this repo's flash-attn (sm_70 drop-in)

Model/data are loaded from the domestic-mirror cache on the AutoDL server:
  /root/autodl-tmp/models/dinov3-h/...   (ModelScope: facebook/dinov3-vith16plus-pretrain-lvd1689m)
  /root/autodl-tmp/models/tiny-imagenet  (HF mirror: zh-plus/tiny-imagenet)
"""
import argparse, glob, json, os, time
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoImageProcessor, AutoModel

USE_FLASH = False

def find_model_dir():
    hits = glob.glob("/root/autodl-tmp/models/dinov3-h/**/snapshots/*/config.json", recursive=True)
    if not hits:
        raise SystemExit("DINOv3 snapshot not found under /root/autodl-tmp/models/dinov3-h")
    return os.path.dirname(sorted(hits)[-1])

def patch_flash_attention():
    import transformers.models.dinov3_vit.modeling_dinov3_vit as m
    from flash_attn import flash_attn_func

    def _flash_forward(self, hidden_states, attention_mask=None, position_embeddings=None, **kwargs):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim)
        cos, sin = position_embeddings
        q = q.transpose(1, 2); k = k.transpose(1, 2)
        q, k = m.apply_rotary_pos_emb(q, k, cos, sin)
        q = q.transpose(1, 2).contiguous(); k = k.transpose(1, 2).contiguous()
        out = flash_attn_func(q, k, v,
                              dropout_p=self.dropout if self.training else 0.0,
                              softmax_scale=self.scaling, causal=False)
        return self.o_proj(out.reshape(B, S, -1)), None

    m.DINOv3ViTAttention.forward = _flash_forward
    print("[flash] patched DINOv3ViTAttention.forward with flash_attn_func")

def preload(split, n, processor, chunk=256):
    ds = split.select(range(min(n, len(split))))
    xs = []
    for i in range(0, len(ds), chunk):
        part = [ds[j]["image"].convert("RGB") for j in range(i, min(i + chunk, len(ds)))]
        xs.append(processor(images=part, return_tensors="pt")["pixel_values"].half())
    x = torch.cat(xs, 0)
    y = torch.tensor([int(v) for v in ds["label"]], dtype=torch.long)
    return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--train-size", type=int, default=4096)
    ap.add_argument("--val-size", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = "cuda"
    model_dir = find_model_dir()
    processor = AutoImageProcessor.from_pretrained(model_dir)
    if USE_FLASH:
        patch_flash_attention()
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora).to(device).train()
    head = torch.nn.Linear(model.config.hidden_size, 200).to(device)

    ds = load_from_disk("/root/autodl-tmp/models/tiny-imagenet")
    t0 = time.time()
    xtr, ytr = preload(ds["train"].shuffle(seed=0), args.train_size, processor)
    xva, yva = preload(ds["valid"].shuffle(seed=0), args.val_size, processor)
    prep_s = time.time() - t0
    print(f"[data] train {tuple(xtr.shape)} val {tuple(xva.shape)} prep {prep_s:.1f}s")

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    n_train = sum(p.numel() for p in params)
    print(f"[lora] rank {args.rank} trainable {n_train/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    def run_step(idx):
        x = xtr[idx].to(device, non_blocking=True)
        y = ytr[idx].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pooled = model(pixel_values=x).pooler_output
            logits = head(pooled)
            loss = F.cross_entropy(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    total = args.warmup + args.steps
    losses = []
    for step in range(total):
        idx = torch.randint(0, len(xtr), (args.batch,))
        l = run_step(idx)
        losses.append(l)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(args.steps):
        idx = torch.randint(0, len(xtr), (args.batch,))
        x = xtr[idx].to(device, non_blocking=True)
        y = ytr[idx].to(device, non_blocking=True)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pooled = model(pixel_values=x).pooler_output
            logits = head(pooled)
            loss = F.cross_entropy(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    peak = torch.cuda.max_memory_allocated() / 2**20

    model.eval()
    correct = total_v = 0
    with torch.no_grad():
        for i in range(0, len(xva), 64):
            xb = xva[i:i + 64].to(device); yb = yva[i:i + 64].to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = head(model(pixel_values=xb).pooler_output)
            correct += int((logits.argmax(-1) == yb).sum())
            total_v += len(yb)
    val_acc = correct / max(total_v, 1)

    import statistics
    res = dict(mode="flash" if USE_FLASH else "sdpa", batch=args.batch, steps=args.steps,
               lora_rank=args.rank, trainable_params=n_train,
               ms_per_step=statistics.median(times),
               samples_per_s=args.batch * 1000.0 / statistics.median(times),
               peak_mib=peak, loss_first=losses[0], loss_last=losses[-1],
               val_acc=val_acc, data_prep_s=prep_s,
               torch=torch.__version__, device=torch.cuda.get_device_name(0))
    print(json.dumps(res, indent=1))
    out = args.out or f"/root/autodl-tmp/code/results_dinov3_{res['mode']}.json"
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)

if __name__ == "__main__":
    main()

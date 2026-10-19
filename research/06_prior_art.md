# 06 · 先例调研（prior-art）：上游对比 · Volta 生态 · 陷阱清单 · 路线建议

> 子代理交付物（prior-art / competitive research）。检索日 **2026-10-19**（所有 URL 与数值均为当日实测/抓取，仅读操作，未改动仓库）。
> 本地基线：`/home/eculid/Documents/Github/flash-attention-v100`（fork of `github.com/mejistus/flash-attention-v100`，上游为 `github.com/ai-bond/flash-attention-v100`），本地 `main` HEAD = **`55d685a`**（"Merge pull request #45 from ai-bond/devel"，2026-09-22），`flash_attn_v100.__version__ = "26.06"`，package 版本 `2.8.3`（API parity 字符串，见 `utils/docs/install.md`）。本地只配置了 `origin=mejistus`，未配置 `upstream` remote。

关键事实速记（详细证据见后文）：

- 上游 ai-bond **有 9 个比本地更新的提交**（最新 `0ababf82e3`，2026-10-14，tag `26.10`）。
- 上游文档自述要求 **CUDA 12.9.1 + torch 2.10.0+cu129**；但 PyTorch 官方口径是 **cu126 构建才是 Volta 的正路**（pytorch/pytorch#157517、dev-discuss 2.11 矩阵）。
- **torch 2.6.0+cu126 wheel 确认包含 sm_70**（构建脚本 arch list `5.0;6.0;7.0;7.5;8.0;8.6;9.0`），我们的 pin 是安全的。
- V100 的现实最优替代是 xformers/SDPA 的 **CUTLASS memory-efficient attention**（sm5x+ 可用，FA 后端 sm80+）；**Triton ≥3.2 在 cc<80 上 tl.dot 退化为 FMA**（无 tensor core），是最大的生态陷阱之一。
- 竞争格局：ai-bond 的仓库是 V100 生态的事实标准之一——1355★ 的 **1CatAI/1Cat-vLLM** 在 README 里直接引用它作为 `FLASH_ATTN_V100` 后端的来源。

---

## 1. 上游对比（ai-bond vs 本地 55d685a）

### 1.1 状态快照（2026-10-19 实测）

| 项 | ai-bond（上游） | mejistus（本 fork 的 origin） | 本地 clone |
|---|---|---|---|
| 默认分支 | `main` | `main` | `main` |
| HEAD | **`0ababf82e3`**（Merge PR #46，2026-10-14T13:30:51Z 提交 / 13:11:29Z push） | `55d685a` | `55d685a` |
| 其他分支 | `devel` 与 `main` **同点**（均为 `0ababf82e3`） | 无（仅 `main`） | — |
| 最新 tag | `26.10` = `0ababf82e3`；历史 `26.06`=`d89800edf6`、`26.04`=`3f43ef732f`、`26.02`=`9070505ee3`、`25.11`=`bd651cec9a`、`v1.2`=`ae5bdc7ced` | 无 tag | — |
| stars | 214 | 0 | — |
| 最近 push | 2026-10-14 | 2026-09-22 | — |

数据源：`https://api.github.com/repos/ai-bond/flash-attention-v100`、`.../branches?per_page=100`、`.../tags`、`.../commits?per_page=50`，以及 `git ls-remote https://github.com/ai-bond/flash-attention-v100.git`（直连 SSL 被重置，经 `https_proxy=http://127.0.0.1:7890` 成功；refs 显示 `refs/heads/main`、`refs/heads/devel` 均指向 `0ababf82e3836c1786d7937c6002670ca4ecda62`）。fork 侧：`https://api.github.com/repos/mejistus/flash-attention-v100/commits?per_page=20` 显示其 HEAD 就是 `55d685a`，**fork 没有独立进展**。

### 1.2 结论：上游有更新，共 9 个提交（55d685a 之后）

按时间顺序（API 返回顺序倒序排列，这里翻正）——逐个核对文件清单来自 `https://api.github.com/repos/ai-bond/flash-attention-v100/commits/<sha>`：

| # | SHA | 日期 | 主题 | 主要文件（+/-） | 评估 |
|---|---|---|---|---|---|
| 1 | `bbed5bd7e8` | 09-30 | Swizzle: fix native mma.h support | `include/swizzle.h` (+6/-1) | **低风险**，单文件小修 |
| 2 | `664c1e08cc` | 10-06 | Kernel: switch dense kernel to **BMHD layout** | `include/kernel.h`(+98/-19)、`mat_mul.h`(+50/-38)、`kernel/forward*.cu`/`backward*.cu`/`forward_kvcache.cu`（约 ±400 行）、`gemm_smem.h`、`softmax.h`、`product.h`、`rotary.h`、`template.h`、`utils/sass/mma_*` 示例内核 | **高风险大改**（~1500 行），但方向（免 permute）正确 |
| 3 | `4e5df293b6` | 10-06 | Interface: fix cast in/out tensors to fp16/fp32 for fp16 kernels | `flash_attn_v100/flash_attn_interface.py` (+306/-230，近乎重写)、`test.py`(+27/-27)、`include/product.h`、`README.md` | **中高风险**；与 #2 强耦合（BMHD 免 permute 后 wrapper 同步重写）；改变输入语义（fp32/bf16 静默 cast 到 fp16） |
| 4 | `653ad746ea` | 10-11 | Template: review GemmTraits by Phase and refactor mma strides | `include/kernel.h`、`mat_mul.h`(+29/-18) | **高风险**（核心模板/步进） |
| 5 | `5f111c007e` | 10-11 | Swizzle: Rework mma_m16n16k16.h swizzle logic and reduce launch_bounds blocks | `include/mma_m16n16k16.h`(+156/-150)、五个 kernel `.cu` 各 ±1、`utils/sass/mma_register/*`（新增 4 个 200+ 行样例）、`compile.sh` | **高风险**（数值/性能路径）；但含可复用的 SASS 样例 |
| 6 | `e20fce4058` | 10-11 | Utils: Rework some incorrect documentation md markdown | `utils/docs/attention.md`、`debug.md`、`install.md`、`volta.md`(+8) | **零风险**（纯文档） |
| 7 | `5cb9f7cdc3` | 10-14 | **Entropy: add fused linear + cross-entropy kernels**（新功能） | 新增 `kernel/entropy.cu`(+1181)、`include/entropy.h`、`flash_attn_v100/cross_entropy.py`、`flash_attn/losses/{__init__,cross_entropy}.py`、`include/mha.h→include/api.h` 重命名、`utils/sass/mma_lce/*`；`setup.py` 等 | **新功能**，与 API parity 目标无关；可选后合 |
| 8 | `3ad659d7fd` | 10-14 | Version change | `flash_attn_v100/__init__.py`、`pyproject.toml`、`setup.py`（各 1 行，版本→`26.10`） | **零风险**（元数据） |
| 9 | `0ababf82e3` | 10-14 | Merge pull request #46 from ai-bond/devel | 上述 1–8 的合并（文件清单为并集） | PR #46 = "Complex fix 14_10_2026" |

### 1.3 提交级详评（并入的价值 vs 风险）

- **#2 + #3（BMHD + 接口重写）必须成对看**：`664c1e08cc` 把 dense kernel 改成直接消费 `(B, M, H, D)`（原来的 wrapper 先 `permute(0,2,1,3)` 再做 kernel）。这解释了两处变化同步发生：kernel 内部布局改为 BMHD 后的矛盾。
  - 我实测 diff 了包装层（本地 401 行 vs 上游 477 行，diff 536 行）：新 wrapper **不再 permute**，改为 `maybe_contiguous`；**输入 dtype ≠ fp16 时直接 `.to(torch.float16)`**（含 fp32/bf16），随后按需 `pad` 到 8 的倍数。语义变化值得警惕：静默 cast 会改变上游 FA2（fp32 走 fp32 kernel）的行为——但它对齐的是本项目的 "fp16 kernels only" 现实。
  - 合并价值：潜在减少一次 permute/copy（对 V100 这类带宽受限卡是真收益），且 wrapper 更接近上游 flash-attn 的搬动方式。风险：两处大改任何一个出问题都会让 vLLM/ComfyUI 类调用点回归；**无 GPU 不可合**。
- **#4/#5（strides 与 swizzle 重做）**：`mma_m16n16k16.h` 是 Volta `m8n8k4` quad-pair 之上封装的 16×16×16 宏 MMA；重做 XOR swizzle + 减少 launch_bounds 分组数属于典型的"性能与边界正确性同调"改动。属于必须 GPU 验证的类型（尤其对 #44/#43 的 edge-mma 场景，它们正是 `06b7189/ec63085/70e1c97` 修复的对象）。
- **#7（LCE）**：`kernel/entropy.cu` 1181 行 + 交叉熵 loss（对齐上游 `flash_attn.losses.cross_entropy`）。对"API drop-in for 2.8.3"有正面意义（上游有 `CrossEntropyLoss`），但工作量大、且 HF 训练路径会同时依赖 kernel 数值。建议优先级低于 API 核心面（fwd/bwd/varlen/kvcache）的稳定性，可延后引入。
- **#6/#8（文档与版本）**：零风险，但**版本号是一个策略选择**：我们的 parity 字符串目前是 `Flash Attention for Tesla V100 v2.8.3`（`pip show flash-attn 2.8.3`）。上游用日历版本（26.06→26.10）+ API 版本 2.8.3 的双轨制，我们已在本地沿用（`__version__=26.06`），同步到 26.10 时需要同时更新 pyproject/setup/init 三处（同 #8 的做法）。

### 1.4 合并策略建议（无 GPU 前提）

- **可直接合（仅编译/文档级）：** `e20fce4058`（docs）+ `3ad659d7fd`（版本元数据）。加一个 compile smoke（`python -m build`/`pip install . --no-build-isolation` 的编译阶段）即可。
- **可考虑单独 cherry-pick：** `bbed5bd7e8`（swizzle.h 6 行）。并行编译 `utils/sass/*` 样例（仓库自带 compile.sh）验证 PTX 可汇编；不改变默认路径行为。
- **条件合（必须 GPU 回归）：** `664c1e08cc`+`4e5df293b6`（BMHD+接口，一步合并）；`5f111c007e`+`653ad746ea`（swizzle/strides+launch_bounds）。合并前要求：diff review + 保留 revert 点 + 用 `test.py` 与新 edge-case 测试（#44 的 Qwen loss 对齐、#43 的 head_dim=128/seq≈3k NaN 复现）做数值回归。
- **暂缓：** `5cb9f7cdc3`（LCE 新功能、无 parity 压力、横跨 Python/C++/build 三面）。
- **流程注意：** 本地没配 upstream remote。将来做"受控合并"时应先 `git remote add upstream https://github.com/ai-bond/flash-attention-v100.git`（或继续用 API/补丁方式），并在合并 commit 里记录来源 SHA。

### 1.5 上游文档/要求确认（含对 Volta 支持声明的质疑）

上游 `utils/docs/install.md`（raw main，与本地 diff 仅链接与文案微调）确认：

- CUDA：**12.9.1**（"Last one available CUDA for Volta"，引 NVIDIA 弃用说明：CUDA 13.0 移除 Volta 的离线编译与库支持）。
- torch：`pip install torch==2.10.0+cu129 --index-url https://download.pytorch.org/whl/cu129`，并称"Install last one PyTorch that's support with 12.9 CUDA"。
- 版本：成功安装后 `flash_attn.__doc__` 应为 `Flash Attention for Tesla V100 v2.8.3`，`pip show flash-attn` → `2.8.3`（即 API parity 语义为 flash-attn 2.8.3）。
- 引用的问题：dev-discuss《Dropping Volta support from CUDA-12.8 binaries for release 2.11》(https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290) 与 pytorch/pytorch#157517。

**【重要疑点】上游推荐的 `torch 2.10.0+cu129` 未必含 sm_70：**

- #157517（标题"[release 2.8-2.9] Delete support for Maxwell, Pascal, and Volta architectures for CUDA 12.8 and 12.9 builds"，2026-01-16 关闭）正文给出 2.8 的实际 arch list：**CUDA 12.6: `5.0;6.0;7.0;7.5;8.0;8.6;9.0`**；**CUDA 12.8: `7.5;8.0;8.6;9.0;10.0;12.0`（已随 2.8 发布到 PyPI）**；CUDA 12.9: `7.5;…;12.0+PTX`。评论确认"把 sm50-70 加回 12.8 会导致 wheel >1GB，超过 PyPI 限制"（PR #157558），atalman 明确："**For Maxwell, Pascal and Volta architecture support please use CUDA 12.6 builds** which are still available for latest releases."
- dev-discuss（2026-01-15，atalman）：2.11 起 cu128 正式移除 Volta；2.11 支持矩阵中 **CUDA 12.6.3 = Maxwell(5.0)…Hopper(9.0)（含 Volta）**，12.8.1/13.0.0 = Turing(7.5) 起。
- 即：**上游安装文档里"检查 Volta"的命令（`get_device_properties(0)` 打印 SM 7.0）只读取设备属性，并不能证明该 wheel 内含 sm_70 内核**。cu129 wheel 是否含 sm_70 存在矛盾信号（社区用户在 2.10.0+cu129 上跑通了 ComfyUI，说明至少部分 2.10 cu129 构建仍可用；但官方 2.8+ 的 12.8/12.9 构建自述已移除）。
- **我们的定位反而更稳**：pin `torch 2.6.0+cu126`，见 §3.9 的证实链。建议我仓文档补一句改正："验证 Volta 请用 `torch.cuda.get_arch_list()` 检查包含 `sm_70`，并跑一个最小 CUDA op；不要用 `get_device_properties` 代替内核存在性检查。"

### 1.6 上游 issue 台账（截至 2026-10-14，共 46 条）

检索方式：`https://api.github.com/search/issues?q=repo:ai-bond/flash-attention-v100&per_page=40`（total=46），标题、状态与部分正文/评论通过 `/issues/<n>` 与 `/issues/<n>/comments` 抓取。

重点 issue 解读：

| # | 状态 | 标题（节译） | 要点与教训 |
|---|---|---|---|
| 45 | PR, closed | Complex fix 22_09_2026 | **本地已含**（55d685a）；`06b7189/ec63085/70e1c97` 修 edge-mma 数值缺陷，直接解决 #44 |
| 44 | ISSUE, closed | 微调后模型"学不动"（Qwen2.5-0.5B, llamafactory, fp16, LR 异常） | ai-bond 让其试 #45；用户 2026-09-23 确认修复后 loss 对齐 ⇒ **本地基线的数值质量已包含该修复** |
| 43 | ISSUE, closed | head_dim=128、seq_len=2925 的有限 Q/K/V 产生 NaN（Anima DiT 训练，torch 2.10.0+cu129，wheel **26.6**） | 用户改用**最新源码重建**后训练恢复正常；维护者质疑该报告为 AI 生成（"clawbot"）。教训：**旧 wheel 与源码不一致是常见"玄学 NaN"来源**；线程未给出确定性 kernel 根因 |
| 36 | ISSUE, 未关 | ComfyUI LTX-2.3 "Input contains (near) NaN/+-Inf" | 最终归因 ffmpeg `aacenc.c` 的报错（≈与 FA 无关）；期间 ai-bond 指出 ComfyUI 插件把 Turing 也打进 patch：**"My FA code didn't support Turing at all! hardcoded to Volta!"** ⇒ sm70-only 是硬事实 |
| 41 | ISSUE, closed | vLLM 0.19/0.21 用户反馈（含 `BLOCK_N_128 160→128` 实验、paged prefill A/B、真实 stdout 争议） | 反馈方仓库 `KumphanartDansiri/v100-vllm-2026`；维护者对 AI 生成补丁要求可复现实证。教训：**tile 常量（BLOCK_N_128）直接关联正确性/性能**，改动必须 GPU 实测 |
| 31 | ISSUE, **open** | vLLM + `TRITON_ATTN` 的 NaN（MiniMax-M2.7-AWQ，1Cat-vLLM 场景） | 讨论横跨 vLLM 与 1Cat 生态；TRITON_ATTN 在 V100 上本身是 FMA 回退（§2.3），问题不必然来自本库 |
| 29 | ISSUE, closed | `AttentionBackendEnum.FLASH_ATTN_V100` + 警告 `No module named 'flash_attn.ops'`（rotary 回退） | **API parity 缺口**：上游 flash-attn 有 `flash_attn.ops.*`，我们的 shim（本地 `flash_attn/` 只有 `bert_padding.py`/`flash_attn_interface.py`/`__init__.py`）没有 ⇒ vLLM/transformers 会静默降级 |
| 27 / 20 / 35 | closed | vLLM 版本约束 / vLLM docker 安装 / Kaggle wheel | 集成需求集中在 vLLM 与 wheel 分发 |
| 19 | ISSUE, closed | "flash_attn_v100_cuda and xformers (shim)" | 生态里存在把本库当作 xformers `memory_efficient_attention` 替身的用法 |
| 16 | ISSUE, closed | RTX 2080 Ti（Turing）能否用 | 不支持；避免对外承诺 Turing |
| 9 / 34 / 33 | closed | `flash_attn_with_kvcache` 特性请求 → PR #34（full features） | KV-cache/paged/varlen 已在本仓库线内实现（本地日志可见 `fb80348`、`690f237` 等） |
| 38 | closed | `cannot import name '_wrapped_flash_attn_backward' from 'flash_attn.flash_attn_interface'` | 兼容层符号缺口（transformers/旧调用约定） |
| 24 / 17 | closed | pyproject 无 torch 依赖 / license 配置报错 | 打包元数据类问题，适合我们做"零风险"打磨 |

**总体规律：** 上游 issue 以"集成类"（vLLM/ComfyUI/HF/wheel）与"AI 生成噪声"为主；**真正的数值缺陷基本集中在 kernel edge 情况，且通过 #45 一类修复 + "用最新源码重建" 收敛**。这意味着：对我们这种无 GPU 的 fork，**最忌讳的是在 kernel 内部做未经 GPU 验证的"优化式重写"**，而集成层/文档/元数据的改进性价比最高。

---

## 2. Volta FA 先例（生态调研）

### 2.1 xformers（CUTLASS memory-efficient attention）——V100 的现实主力

- **方案**：非 FA 的 CUTLASS 实现，`memory_efficient_attention` API，可跑 FW/BW；xformers 维护者 danthe3rd 明确："cutlass 后端比 FlashAttention 稍慢，但支持更多设备（**任何 P100 以上的 GPU**）"（https://github.com/facebookresearch/xformers/issues/950）；另一讨论："已测试 6.0+，理论上 5.0+，底层依赖 CUTLASS"（https://github.com/facebookresearch/xformers/discussions/689）。
- **限制**：tests 以 `sm70_or_better_only` 等 marker 区分后端（https://github.com/facebookresearch/xformers/blob/main/tests/test_mem_eff_attention.py）；cutlass 后端在 SM7.0+ 需要 **128-bit 对齐**，最大 headdim 65536，支持 f32/f16（bf16 需 sm80+）（https://deepwiki.com/facebookresearch/xformers/2.1.2-cutlass-backend）；历史坑：wrong-arch kernel 报错（"fmha_cutlassF_..._sm80 is for sm80-sm90, but was built for sm70"，xformers#692）、`attn_bias` 未对齐报错（xformers#813，V100 + 43 长度 mask 实测）。
- **性能锚点**：AlexWortega/flash-attn-volta 的实测把 xformers CUTLASS 当作 V100 上限：fp16 `(1,1024,16,64)` causal ≈ 30.1 TFLOP/s、`(1,4096,16,64)` causal ≈ 58.3 TFLOP/s、head_dim=128 ≈ 24–37 TFLOP/s（https://github.com/AlexWortega/flash-attn-volta）。**这就是 V100 上"FA 类注意力"的现实带宽（30–60 TFLOP/s 量级）**。
- **教训**：任何 V100 kernel 改进都应对标 xformers CUTLASS 而不是 A100 上的 FA 数字；我们的 kvcache/backward 若能量级贴近它即为胜利。

### 2.2 PyTorch SDPA（mem-efficient backend）

- **方案**：SDPA 三个后端中，`MEM_EFFICIENT` 即为 xformers 同源的 CUTLASS 内核（PyTorch 内建 `_efficient_attention_forward/backward`）。官方博客给出硬性依赖："**The CUDA architecture level must be sm5x or better for the mem_efficient kernel, and sm80 for Flash Attention**"（https://pytorch.org/blog/accelerated-pytorch-2/）。
- **限制**：Flash SDPA 后端在 V100 不可用；SDPA 的 GQA（`enable_gqa`）目前仅 FA/math 后端支持（2.6 文档），V100 上 GQA 只能走 math/手工 repeat；fp32 输入下 mem-efficient 可用（HF 文档）。
- **历史**：pytorch/pytorch#91994 引入 FA 主分支时明确"Cutlass 分支与 main 分支都不支持 Volta；**SDPA 的 Volta 支持来自 mem-efficient kernel**"（drisspg 2023）。HF 侧：issue #28861 中官方回复"V100 请用 `attn_implementation="sdpa"`"（https://github.com/huggingface/transformers/issues/28861）。
- **教训**：V100 用户默认路径是 SDPA/mem-efficient；本库的差异价值在于 **API 同名 drop-in + 更快/更省显存**。若我们的 kernel 无法同时给出正确性与性能优势，"换回 SDPA"是用户的最优解——这要求我们文档里如实标注适用场景。

### 2.3 Triton 路线在 Volta 上的现状（重要陷阱）

- **MMAv1 已被弃用**：triton-lang/triton#5066《[BACKEND] Deprecate MMAv1; fallback to FMA for computeCapability < 80》2024-11 合并；README 的最低支持硬件随之提高到 **cc 8.0+**。维护者在 2025-03 的评论："支持不会破坏，但 pre-Ampere 只会 best-effort 且不再测试；欢迎修 bug 但**不太可能接受复杂的性能补丁**"（https://github.com/triton-lang/triton/pull/5066）。
- **版本耦合**：torch 2.6.0 在 Linux x86_64 上 `requires triton==3.2.0`（PyPI 元数据 https://pypi.org/pypi/torch/2.6.0/json）。即 **torch 2.6 + triton 3.2 在 V100 上 `tl.dot` 已退化为 FMA（SIMT）路径**，`torch.compile`/TRITON_ATTN 类负载失去 tensor core。
- **历史先例**：Triton 的 fused attention 教程最初为 Volta 撰写，但后来 gate 在 A100（triton#616 "Making fused attention work with GPUs other than A100"、openai/triton#1279）；`AlexWortega/flash-attn-volta` 为保留 V100 tensor core，特意锁 **Triton 2.3**，并记录 V100/Triton 2.x 陷阱：block size 必须整除 head_dim（推荐 128/64/32）、`tl.where(mask, val, -inf)` 在 Volta 需要 fp32 cast（否则用 -1e9 代替）、**96KB SRAM 预算计算**、`tl.exp/exp2` 本身可用。
- **vLLM 侧佐证**：vLLM issue #47019 发现 V100 上 MoE decode 的默认 `BLOCK_SIZE_K=128` 严重 spill，"最初怀疑 `num_stages`，因为 **Volta 缺少 cp.async**；实际根因是 K 维 tile 对寄存器/smem 的压力"，`BLOCK_SIZE_K=64` 修复且输出 bit-identical（https://github.com/vllm-project/vllm/issues/47019）。
- **教训（对接我们的路线）**：**不要在 Triton ≥3.0 上重写本库**；`mma.sync m8n8k4` PTX（本库现有路线）+ 手工 smem swizzle 才是 V100 的正道。tile 常量（BLOCK_N/BMHD/stages）的调整必须对应寄存器/smem 预算表。

### 2.4 vLLM 生态：V100 注意力后端的真实图景

- 当前 vLLM 官方文档（https://docs.vllm.ai/en/latest/design/attention_backends/）：标准注意力后端中 **`FLASH_ATTN` 要求 CC ≥ 8.0**（FA2），`TRITON_ATTN` 任意 CC（但见 §2.3 的 FMA 退化），`FLEX_ATTENTION` 任意 CC；历史上 V100 依靠 **XFORMERS 后端**（vllm PR #877 时期即有"FA2 支持不了 V100，需要回退 xformers cutlass"的讨论；issue #6173 记录 V100 + Gemma2 走 flashinfer 失败、需绕行）。
- **官方弃养时间线**：vLLM v0.20.0 起文档最低 CC 改为 7.5（用户实测：v0.17–0.19 还列 V100，但 AWQ 路径强制 `get_min_capability()=75`，SM 7.0 直接被拒；SGLang 因 FlashInfer（≥7.5）彻底无缘）（https://github.com/vllm-project/vllm/issues/43561 的 2026-08-12 评论）。vLLM 0.19 被视为"最后一个对 V100 广泛实用的版本线"（#47019 正文）。
- **社区前行**：
  - **1CatAI/1Cat-vLLM**（1355★，V100/SM70 专项 vLLM fork，2026-10-17 仍在推）：`FLASH_ATTN_V100` 后端 + `flash_attn_v100`(含 `flash_attn_v100_cuda`/`paged_kv_utils`) 运行时扩展；**README 致谢直接引用 [flash-attention-v100](https://github.com/ai-bond/flash-attention-v100)**（https://github.com/1CatAI/1Cat-vLLM）。其工程博客给出 V100 长上下文注意力优化数字（17.92→≈71 TFLOP/s；128-bit 加载改造：L1 请求 -41.5%、kernel 时长 -22.95%）——**这些正是"向量化/宽加载"类无 GPU 可规划、GPU 才可验证的改进方向**。
  - `zhinianqin/flash-attention-v100`（69★，2026-05）：把 FA2 移植到 SM70，宣称 q8k 场景比 `TRITON_ATTN` 快 580%，配合 vLLM 0.19.1 只需一行补丁（`supports_compute_capability` 8.0→7.0），依赖 torch 2.10.0+cu128/CUDA 12.8（https://github.com/zhinianqin/flash-attention-v100）。
  - `KumphanartDansiri/v100-vllm-2026`：围绕本库的 vLLM 0.19/0.21 适配与反馈包（见上游 issue #41）。
- **教训**：我们的接口/符号命名（`flash_attn_v100_cuda`、paged 相关、`torch.ops.flash_attn_v100`、`flash_attn_2_cuda`）**已经被下游 vLLM fork 生态依赖**，属于"公共契约"，改动要向后兼容；decode/paged 场景既是需求热区，也是（如上）最需要 GPU 验证的区域。

### 2.5 unsloth

- 官方 requirements 页面自述："**Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20 & 50…)**"，并要求系统具备 `xformers`、`triton`（https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements）。
- 但历史上（2024）其 FA2 路径在 V100 上是坏的：issue #496 显示其推荐安装脚本把 <CC8 的用户引向 `xformers<0.0.26`，xformers 报错列表里 `flshattF requires capability > (8,0)`、`cutlassF` 未构建；issue #948 的经典提示："**Your Flash Attention 2 installation seems to be broken… We shall now use Xformers instead**"（https://github.com/unslothai/unsloth/issues/948）。2026-08 他们还合并了"fp32 加载模型不要选 FA2"的修复（PR #9110）。
- **结论**：unsloth 的 V100 支持 ≈ xformers/triton 路径，**并不提供 V100 的 FA2 内核**；不存在可直接引入的"unsloth V100 FA kernel"。我们能借鉴的是其"自动降级 + 明确告警"的工程姿态。

### 2.6 HuggingFace / Dao-AILab 官方立场

- Dao-AILab/flash-attention README：FA2 CUDA 可用硬件 = "**Ampere, Ada, or Hopper**（A100/3090/4090/H100）… Turing（T4, RTX 2080）请用独立的 flash-attention-turing"；requirements 指向 CUDA ≥12.3、PyTorch ≥2.2（https://github.com/Dao-AILab/flash-attention）。V100 连 Turing 的门都没有。
- issue #68（2022，至今 open）里 tridao 给出根因："Volta 的 tensor core 乘的是 4 组 8×4/4×4（`mma.sync.aligned.m8n8k4`），Turing/Ampere 是 16×8×8/16×16×8；**输入在 shared memory 的布局不同**，这是最难的部分；CUTLASS 对 Volta 一直是独立代码路径"，并推荐 xformers（https://github.com/Dao-AILab/flash-attention/issues/68）。issue #1760（2025-07）"V100 能否用 FA"→关闭（无解）。
- HF transformers：`flash_attention_2` 在 V100 直接抛 "FlashAttention only supports Ampere GPUs or newer"；官方建议退到 `sdpa`（issue #28861）。2026-08 的 PR #48024 还专门把 kernels 库的 FA 回退 gate 在 CC≥8.0（此前 T4/V100 会在生成阶段崩溃）。
- **教训**：所有主流栈在 V100 上都把 FA 排除在外 ⇒ 本库的护城河就是"唯一活跃的 sm70 专实现"，因此**正确性声誉 > 激进优化**。

### 2.7 其他 "flash attention volta/v100" GitHub 项目一览

来源：GitHub repo search（`flash+attention+v100`、`volta+flash+attention`、`flash+attention+sm70`）与各项目 README/API。

| 项目 | ★ | 最近活动 | 方案 | 局限（自述/实测） | 对我们的启示 |
|---|---|---|---|---|---|
| `ai-bond/flash-attention-v100`（本上游） | 214 | 2026-10 | 手写 CUDA，PTX `mma.sync.m8n8k4` + quadpair 封装 `mma_m16n16k16`，自研 swizzle；API 对齐 FA 2.8.3；kvcache/varlen/backward 全 | 仅 sm70；Turing 不支持；无 GPU 时代码质量依赖作者自有硬件 | 事实标准之一；1Cat-vLLM 直接引用 |
| `ZRayZzz/flash-attention-v100` | 80 | 2024-02 | 独立实现：按 CUTLASS 教程 + FA2 论文自写（CUTLASS 头 + setup.py 手改） | 自述"forward 比 PyTorch 快约 40%、backward 慢约 20%，两者抵消"；无边界处理，**要求 right-pad 到 32 的倍数**；2024 后停更 | 早期先例；提醒：没有 backward 优化与边界处理的"FA" 不值钱 |
| `zhinianqin/flash-attention-v100` | 69 | 2026-05 | 面向 vLLM 0.19.1 的 SM70 移植 + 一行 vLLM 补丁 | 需求 torch 2.10+cu128（注意 §1.5 的 cu128/Volta 疑点）；性能宣称未独立复核 | vLLM 集成是兵家必争；我们的符号契约要稳 |
| `sirCamp/flash-attention-legacy` | 27 | 2026-04 | 手写 CUDA：**Volta 用 WMMA 16×16×16**、Pascal 用 half2；提供 HF `attn_implementation="flash_attn_legacy"`；支持 MHA/GQA/MQA + varlen；decode 回退普通注意力 | fp16 仅 d=64/128；无 paged/kvcache；prefill-only 策略（q_len=1 不跑 FA） | "HF 集成 + varlen + GQA" 的另一条产品化路线；WMMA 路线与我们的 mma.sync 路线形成对照 |
| `AlexWortega/flash-attn-volta` | — | 2026 前后 | Triton 2.3 手写 FA fwd+bwd，对 xformers CUTLASS 做 perf-steal/autotune | seq=1024 时仅 33–55% xformers；无 fp8/bf16/varlen/dropout/ALiBi；GQA 靠 repeat_kv | 最诚实的 V100 基准数据来源；其 pitfalls 清单直接可抄 |
| `1CatAI/1Cat-vLLM` | 1355 | 2026-10 | vLLM fork：TurboMind SM70 内核 + `FLASH_ATTN_V100`（引用本上游）+ AWQ | 实验特性（DFlash/MTP）尚不稳定；维护者为 V100 经销商生态 | 我们的最大潜在用户；decode 路径与 128-bit 访问优化样本 |
| `ssui/flash-attention-turing` | — | — | Turing（sm75）方向（Dao README 钦点） | 与 sm70 无关 | 不要混合 Turing/sm70 承诺 |
| ComfyUI 节点（`FearL0rd/ComfyUI-Flash-Attention_v100`、`NetVoobrazhenia/ComfyUI_Flash-Attention_v100`） | 31 等 | 2026 | 把本库 patch 进 ComfyUI 的 FA1/FA2 路径 | 曾把 Turing 误纳入 patch；报错归因易误伤本库（issue #36） | ComfyUI 是最活跃的国内用户面；文档要给出"如何验证归属"的指引 |
| `Mengarr/flash-attention-v100` 等 fork | 0 | 2026-06 | ai-bond 的 fork，贡献过 kvcache guard 修复（`4c439ee` 致谢 f31657c） | — | 上游会吸收社区修复；我们反馈上游有路径 |
| `ADA-quart/bonsai2-27b-v100-optimizations`、`ArjunSom9/VoltaFlash` | 0–1 | 2026 | 垂直场景/学习型内核 | 单点、未验证 | 可忽略（除证明 V100 长尾需求仍在） |

---

## 3. Volta 专项陷阱清单（公共来源交叉验证）

### 3.1 fp16 累加与中间态溢出（重中之重）

- **事实**：FP16 最大有限值 65504；`exp(x)` 在 x>11.09 即溢出（fp16），长序列的 `sum(exp)` 也可越界（https://mlinterviewnotes.com/courses/inference/03-attention-kernels/03-01-naive-attention-and-online-softmax.html）。
- **官方口径**：tridao 在 Dao-AILab/flash-attention#543 明确："**Accumulation is done in fp32 for numerical precision and numerical stability. Fp16 might get inf if e.g. Q @ K^T has entries larger than 65k.**"；同 issue 中 4090 上的实验显示 fp16 累加可让吞吐近乎翻倍（Ada 消费卡特性），但**实测真实模型数值确有超过 65504 的分布，必须 clamp inf**，且作者也承认有精度损失（https://github.com/Dao-AILab/flash-attention/issues/543）。
- **Volta 硬件特性**：V100 Tensor Core 原生支持 FP16 输入 + **FP16 或 FP32 累加**（NVIDIA《Programming Tensor Cores in CUDA 9》：每个 TC 每时钟 64 FMA，FP16 乘 + FP32 累加；arXiv 2006.12645 亦注明".f16/.f32 accumulator 两种 mma.m8n8k4 均存在"）。**没有必须用 fp16 累加的理由**——本库文档/回应（issue #36 的 ai-bond 回复）称内部计算为 FP32 导入导出 FP16。
- **其他公共防御手法**（可借鉴）：llama.cpp `fattn-common.cuh` 用 `HALF_MAX_HALF = 65504/2` 初始化 KQ max 以避免 `-INF` 相减 NaN、`SOFTMAX_FTZ_THRESHOLD=-20` 把极小 exp 归零、`FATTN_KQ_MAX_OFFSET` 偏移 max 以防 VKQ 累加溢出（https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh）。
- **对我们的动作**：任何触及累加器类型/缩放/最大值偏移的改动 = 高风险；KVCache 里 **V 值用 fp16 存储** 时也受 65504 约束（long-context 权重和会贴近上限）。issue #43（hd=128/seq≈3k 的 NaN）虽未定论，但"K 维累加必须 fp32 + 逐块 max 重缩放 + 溢出 clamp"应写入回归测试目标。

### 3.2 exp/softmax 精度与 `--use_fast_math`

- **MUFU 瓶颈**：tridao 在#1225 指出 `exp` 走 MUFU，"每时钟约 16 条，是加减乘的 4–8 倍慢"；FA 全系用 `exp2f` + `scale·log2(e)` 融合成 FFMA（https://github.com/Dao-AILab/flash-attention/issues/1225、DeepWiki online-softmax）。
- **FMA 融合的数值坑**：PyTorch 特意引入 `UNFUSE_FMA`，避免 `tensor*scale - max_scaled` 被融合后与参考实现产生 subnormal 差异（https://github.com/pytorch/pytorch/issues/121558）。
- **本库现状**：`include/softmax.h` 直接使用 `__expf`（MUFU.EX2 路径）并把指数参数下限 clamp 到 **-80.0f**（第 93/99-102/148/282-285 行）——即已经显式选择"快 + 有界"，并未依赖全局 `--use_fast_math`。
- **风险结论**：开启全局 `-use_fast_math` 会顺带把除法/开方/去 normal 的行为改掉，结果在不同编译选项下漂移；调试时又会被误当成 kernel bug。**保持"显式 intrinsic + 固定 clamp"策略**；如需精度对照，用"未优化版本编译一份"而不是切换全局 fast-math。（精度参考：CUDA Math API 内在函数表 `__expf` 误差 ~2 ulp。）

### 3.3 无 `ldmatrix` 与 bank conflict（Volta 的 smem 之痛）

- **硬事实**：`ldmatrix` 需要 `.target sm_75` 以上（本库 `utils/docs/volta.md` 记录编译报错原文："Feature 'ldmatrix' requires .target sm_75 or higher"），所以 Volta 上的 K/V 片段装载**必须手写**，XOR swizzle 与 padding 是唯一手段。
- **布局根因**：tridao 在 #68 解释：m8n8k4 与 m16n8k8 需要不同的 smem 布局；"**shared memory load/store is the trickiest part**"（https://github.com/Dao-AILab/flash-attention/issues/68）。
- **上游实践**：2026-06 起 ai-bond 密集合入 swizzle 相关提交（`42e95fe` "add swizzle load/store bank conflicts reduce"、`3d2f48f`、`70e1c97` "accumulator load/store as per-row/col XOR"、`4e2273a` "shared-memory ld/st helpers"），并且作者在 issue 里报告"**swizzle 带来最高 30% 计算加速**"（issue #36 评论）。
- **对我们的动作**：swizzle/取出模式属"静态可对照、运行才知收益"类改动；可以准备（并且上游后续的 `5f111c007e` 重做正说明这一块一直在动），但**不要在没有 V100 的情况下动默认布局**。

### 3.4 WMMA vs `mma.sync` 的取舍

- 本库 `volta.md` 的立场（原文）：`nvcuda::wmma` 在 CUDA 12.4 上只能生成 m16n16k16 风格路径，**无法直接生成 `mma.sync.m8n8k4`**，因此"用 wmma 在 Volta 上做事不可行"，本库选择 inline PTX m8n8k4。（该文对 wmma 的描述偏绝对，但从工程结果看结论成立：wmma 片段布局固定、指令组合不透明。）
- 独立证据：微基准论文《Dissecting Tensor Cores》指出 Volta 上 **每条 `wmma.mma` 会被编译为多条 `HMMA.884`**，而直接 `mma.m16n8k16` 在 Turing/Ampere 上对应单条 HMMA；"追求性能应使用新编程接口"（https://doi.org/10.1109/tpds.2022.3217824 与 arXiv 2006.12645）。
- 对照：`sirCamp/flash-attention-legacy` 用 WMMA 16×16×16 也做成了 V100 可用实现（prefill-only），说明 wmma 能工作但难榨性能；本库的 mma.sync + quadpair 封装更接近 xformers/cutlass 的做法。
- **动作**：维持 mma.sync 路线；新代码不要引入 wmma；对 `mma_m16n16k16.h` 这类封装，改动必须配 SASS 样例（仓库 `utils/sass/*` 正是为此存在，compile.sh 可离线跑 `nvcc -ptx/-cubin`）。

### 3.5 96KB smem、寄存器与 occupancy

- V100：每 SM 96KB shared memory（opt-in 需 `cudaFuncAttributeMaxDynamicSharedMemorySize`）、64K 32-bit 寄存器、最大 2048 线程/SM（CUDA C Programming Guide compute capability 表：https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities）。
- 布局预算示例（AlexWortega RESEARCH.md 的算法）：`(BLOCK_M·D + 2·BLOCK_N·D)·2B + acc/m/l`，例如 BM=128/BN=64/D=128 ⇒ 64KB，把 KV 双缓冲卡死；这直接约束我们的 tile/num_stages 组合。
- 相关信号：上游 2026-06 加了 `smem occupancy calculator`（`5d5707b`）与"reduce launch_bounds blocks"（`5f111c007e`）——都在处理"96KB 内放置多阶段流水"的问题；vLLM #47019 的 Volta spill 案例同样源于寄存器压力。
- **动作**：把 tile/launch_bounds/共享内存计算纳入静态检查（编译期 `static_assert` + 占用表脚本），杜绝"越界才在 GPU 上炸"的改动。

### 3.6 Causal masking、边界与尾部

- FA2 的 causal 效率来自**整块跳过 + 行级 mask**；上游/本库在尾部与 mask 上修过多次（本地日志：`8a86213` "Fix compile/runtime tail process"、`4b0bd48` "prevent mma from reading garbage from sP for varlen/kvcache"、`e51b63c`、`35a3777` "direct P stores + correct half-stride"、`ec63085` "single-pass P/mask store"）。
- 已知数值边界：
  - 全 mask 行 `(-inf) - (-inf) = NaN`：FA cute 后端用 `check_inf` 把 `row_max==-inf` 置 0（DeepWiki）；本库 `softmax.h` 用 `NEG_INF` 判断直接给 0 概率（第 282-285 行）——两者是同一类防御，回归测试要覆盖"整行被 mask / 空 KV 段"。
  - 尾部 tile 未清零会读脏数据（上游 `4668250`/`9b78166` "refills with 0" 就是在修这个）。
- **动作**：causal/窗口/softcap/ALiBi 的组合矩阵测试（数值 tolerances）只能在 GPU 上做；离线侧可做的是保持断言与调试宏（仓库 `utils/docs/debug.md` 的 `__CHECK_ERRORS`/`__CHECK_INIT`）与新测试用例的**代码就绪**。

### 3.7 KV-cache / decode（V100 最现实的痛点）

- 背景：vLLM 从 FA 2.5.0 开始接入 paged KV（vllm PR #3010）；解码阶段 attention 是"读带宽受限 + 低算术密度"，V100 的 32GB 版本尤其受限于显存带宽。
- V100 专属问题：`TRITON_ATTN` 的 paged 解码在 FMA 回退下更慢；vLLM #47019 的 `BLOCK_SIZE_K` spill；1Cat-vLLM 报告 128-bit 加载使 L1 请求 -41.5%、kernel -22.95%（paged-KV 分区内复用 Page ID、合并 half8 转换组、一次对齐 128-bit load，软/归约顺序不动）——**这类"存取向量化"是 V100 decode 的主要收益来源**。
- 本库现状：已有 `fwd_kvcache`（本地 `flash_attn_interface.py:366`、kernel `forward_kvcache.cu`），issue #9/#34 完成了 full-features 支持；`4c439ee` 修了空 KV 的 stride crash（与 Mengarr fork 的 `f31657c` 交叉致谢）。
- **动作**：decode 相关改动（split-K、paged 地址计算、128-bit 向量化）**必须 GPU 验证**；离线能做的是把地址计算/边界条件写成可单元测试的纯函数，并把 "T_NEW=0"、"空段"、"非连续 cache" 等用例固化。

### 3.8 GQA / MQA

- 上游 FA2：GQA/MQA 内核 sm80+；SDPA 的 `enable_gqa` 仅 FA/math 后端支持（2.6 文档）。
- V100 生态做法：xformers CUTLASS 支持 MQA/GQA（带 bias）；AlexWortega 在 HF patch 层 `repeat_kv`（以显存换简单）；本库在 backward/varlen 用"按 head 分组 + 每 head launch"实现（`690f237`）；1Cat-vLLM 做了 **GQA-packed 宽 GEMM**（≈60.8 TFLOP/s）。
- **动作**：GQA 性能优化（packed GEMM）需要大改与 GPU 验证；短期保持"正确优先"。文档需说明 GQA 在 V100 上的开销来源。

### 3.9 torch / wheel 侧的 sm_70 现状（本任务的关键问题）

**结论：torch 2.6.0+cu126（x86_64）包含 sm_70 内核；我们的 pin 有完整证据链。**

1. **官方构建脚本**（tag `v2.6.0`，`.ci/manywheel/build_cuda.sh`）：
   - 第 60 行：`TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6"`；
   - 第 62–67 行：`12.6` 分支对 x86_64 追加 `;9.0` ⇒ **`5.0;6.0;7.0;7.5;8.0;8.6;9.0`**（aarch64 例外，仅 9.0）。
   - 链接：https://github.com/pytorch/pytorch/blob/v2.6.0/.ci/manywheel/build_cuda.sh
2. **wheel 存在性**：`https://download.pytorch.org/whl/cu126/torch/` 索引中 `torch-2.6.0+cu126-*manylinux_2_28_x86_64.whl` 覆盖 cp39–cp313。
3. **依赖版本**：PyPI 元数据 `torch==2.6.0` → Linux x86_64 依赖 `triton==3.2.0`（§2.3 的 FMA 回退即绑定此版本）与 `nvidia-cudnn-cu12==9.1.0.70`（cuDNN 9.1 仍支持 Volta；cuDNN 只在新版本里弃用 Volta，参见 pytorch#157517 讨论中 ngimel 的解释）。
4. **官方口径**：pytorch#157517 显示 2.8+ 的 12.8/12.9 构建删除了 ≤sm70（wheel 体积；PR #157558），并明确"老架构请用 **CUDA 12.6 builds**"；dev-discuss 2.11 矩阵进一步确认 **12.6.3 = Maxwell…Hopper（含 Volta）**。也就是说：**"2.6/2.7 掉 sm_70" 的说法并不成立——掉的是 2.8+ 的 cu128/cu129 构建；cu126 一直保留**。
5. **实践校验命令**（建议写入我们文档/CI 说明）：
   `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"` → 应含 `sm_70`；再跑 `torch.ones(1,device='cuda')*2` 冒烟即可排除"能 import 但无内核"的情况。
6. **顺带结论**：上游推荐 `torch 2.10.0+cu129` 存在风险（§1.5）；我们锚定 cu126 反而与 PyTorch 官方对 Volta 的指引一致。**不要为了追平上游文档而升级到 cu128/cu129 构建，除非先验证 arch list。**

### 3.10 其他零散但致命的点

- **V100 无 bf16/tf32**：bf16 会被 cast 到 fp32 计算，性能骤降（社区问答：https://genai.stackexchange.com/questions/2242/why-is-vllm-inference-slow-on-v100-gpus-with-bitsandbytes-quantized-models）；本库与生态先例都只做 fp16/fp32。
- **Turing 不在支持范围**：ai-bond 明言代码"hardcoded to Volta"（issue #36 评论）；上游 issue #16 已拒绝 RTX 2080 Ti 类需求。文档要避免模糊措辞。
- **`flash_attn.ops` 缺口**（issue #29）：vLLM/transformers 会尝试 `flash_attn.ops.rotary` 等，缺模块时静默回退 native rotary。纯 Python shim 即可消除告警/行为差异，属无 GPU 可做。
- **兼容符号缺口**（issue #38）：`_wrapped_flash_attn_*` 等旧私有符号被部分框架引用；同理属于 shim 层面可补。
- **确定性/可复现性**：多流/原子 add（backward 的 dQ/dK 归约）与 split-K 会影响 bit-level 复现；`deterministic=True` 在上游/本库均是"半支持"（README 自述 "not fully supported"）。文档如实标注。
- **Windows/ComfyUI**：上游 issue #32/#40/#21/#15 显示 Windows 与 ComfyUI 是常见使用面；我们的构建脚本与文档若只针对 Linux，要写明。

---

## 4. 推荐路线（按"无 GPU 可验证性"排序）

### 4.1 A 类：无 GPU 可安全推进（建议列入近期计划）

1. **文档与元数据同步**：cherry-pick `e20fce4058`（docs）+ `3ad659d7fd`（版本 26.10；若坚持 2.8.3 parity 字符串则只同步不改变 API 声明）；同时新增一节"sm_70 内核存在性验证"（`get_arch_list()` + 冒烟），修正上游文档的 `get_device_properties` 陷阱。
2. **API-shim 补全（纯 Python，零 kernel 风险）**：
   - 增加 `flash_attn/ops/` 兼容模块（至少 rotary 的正确实现/代理，可选 layer_norm/fused_dense 的 fallback），消除 issue #29 的降级告警；
   - 审计符号面：`flash_attn_v100_cuda`（fwd/bwd/varlen_fwd/varlen_bwd/fwd_kvcache）、`flash_attn_2_cuda` symlink（setup.py 已做）、`torch.ops.flash_attn_v100`、`flash_attn_v100.__version__`，写一个"契约清单"测试（import 级，无需 GPU 写 kernel 调用断言）。
3. **构建与静态检查硬化**：CI 增加 cu126+torch2.6.0 的 compile-only 任务；编译期断言（`static_assert`）覆盖 smem 预算、对齐、head dim 取模；用仓库自带 `utils/sass/*/compile.sh` 对新改动生成 PTX/SASS 做人工核对。
4. **零风险小修**：`bbed5bd7e8`（swizzle.h 6 行）；对 `include/swizzle.h`、`mma_m16n16k16.h` 增加"仅重排索引"的恒等测试（CPU 模拟 XOR 地址计算，验证与朴素地址一致的映射关系）——无 GPU 也能保证索引不越界/不冲突的数学正确性。
5. **测试弹药准备**（为将来 GPU 窗口做储备）：把 #43（hd=128, s=2925, fp16）与 #44（Qwen 微调 loss 对齐）写成可复现脚本；准备与 xformers CUTLASS/SDPA 的数值对比 tolerance（fp16 输出 rtol/atol 经验值），一旦拿到 GPU 即可开跑。
6. **存取向量化的"可规划项"**：对照 1Cat-vLLM 的 128-bit 加载案例，评估现有 kernel 中 K/V 全局加载的宽度（`float4`/`half8` 对齐条件），把改造设计成编译期开关；默认路径不动。
7. **（可选）LCE 特性预研**：阅读 `kernel/entropy.cu` 设计（不合并），评估作 parity 补全的距离；标注"需 GPU 数值验证"。

### 4.2 B 类：仅可做静态准备，合入默认必须 GPU 验证

- `664c1e08cc` + `4e5df293b6`（BMHD dense 布局 + wrapper 重写）：一步合并、单独分支、保留 revert；验证项：`test.py`、#43/#44 复现、vLLM dense 调用点、dtype cast 语义变化对 fp32/bf16 用户的影响。
- `5f111c007e` + `653ad746ea`（swizzle/strides/launch_bounds）：验证项：数值 tolerance、edge-mma 用例（#44 系列）、寄存器/smem 占用未越界。
- 任何 tile 尺寸/阶段数改动（参考 #41 的 `BLOCK_N_128` 教训与 #47019 的 spill 教训）。
- causal/window/softcap/alibi/kvcache 的逻辑改动；GQA packed-GEMM 优化。

### 4.3 C 类：明确避免

- **Triton ≥3.0 重写**（cc<80 无 tensor core；仅 Triton 2.x 路径可考虑，但已停止维护、与 torch 2.6 不兼容）。
- 全局 `-use_fast_math`；把"与参考实现的 bit 级一致"当验收标准（FMA 融合/FTZ 本来就允许差异——用 tolerance 与统计口径验收）。
- 以 bf16/tf32 为一等公民的设计；任何 Turing（sm75）兼容承诺。
- 在无 GPU 情况下把上游 #46 的内核大改（BMHD/swizzle/strides）直接设为默认行为；或者把上游文档的 `torch 2.10.0+cu129` 建议原样照抄（见 §1.5/§3.9）。
- 依赖"上游一合了之"的 rebase 策略：上游仍处于高频、大颗粒度的 kernel 演化期（PR #45、#46 连续两轮 "Complex fix"），我们应保持"小步、可回退、仅合已验证子集"。

### 4.4 若拿到 GPU 的验证顺序（建议）

1. 基线自检：`test.py` 全绿 + `torch.cuda.get_arch_list()` 含 sm_70；
2. #43/#44 复现用例（数值边缘）；
3. 与 SDPA(mem-efficient)/xformers CUTLASS 的 FWD/BWD 数值对比（tolerance 表）；
4. kvcache/paged/varlen 集成（vLLM 0.19 或 1Cat-vLLM 路径）；
5. 性能对照（vs xformers CUTLASS 与自身基线，序列 1k/2k/4k × causal/non-causal × hd 64/128）。

---

## 5. 速览（≤15 行）

1. **上游有更新（y）**：ai-bond `main`=`devel`=`0ababf82e3`（2026-10-14，PR #46，tag 26.10），比本地 `55d685a` 多 **9 个提交**：`bbed5bd7e8`(swizzle.h)、`664c1e08cc`(dense→BMHD)、`4e5df293b6`(interface 重写)、`653ad746ea`(GemmTraits/strides)、`5f111c007e`(swizzle+launch_bounds)、`e20fce4058`(docs)、`5cb9f7cdc3`(Entropy/LCE 新特性)、`3ad659d7fd`(版本 26.10)、`0ababf82e3`(merge)；mejistus fork 无独立进展。
2. 合并建议：可无 GPU 合 = docs+版本（#6/#8）+ swizzle.h 小修（#1）；BMHD+接口（#2+#3）与 swizzle/strides（#4+#5）**需 GPU 回归**后整包合；LCE（#7）暂缓。
3. 上游文档要求 CUDA 12.9.1 / torch 2.10.0+cu129；**但 cu128/cu129 自 2.8 起删除 ≤sm70**（#157517、dev-discuss 2.11 矩阵），仅 cu126 保留 Volta——上游的"Volta 检查命令"（get_device_properties）不能证明内核存在。
4. **torch 2.6.0+cu126 含 sm_70（证据）**：v2.6.0 `build_cuda.sh:60-67` → `5.0;6.0;7.0;7.5;8.0;8.6;9.0`；cu126 索引存在 cp39–cp313 wheel；PyPI 依赖 `triton==3.2.0`/`cudnn 9.1.0.70`。
5. 头号陷阱：fp16 累加/中间态溢出（65504；fp32 累加是默认，见 FA #543）；`__expf`+clamp(-80) 与全局 fast-math 的漂移；无 `ldmatrix` 的 smem 布局（swizzle 收益曾达 ~30%）；96KB smem/寄存器预算；masked 行 `-inf` 与尾部脏数据；KVCache 解码的 spill/带宽（BLOCK_SIZE_K、128-bit 加载）。
6. 生态：Triton ≥3.2 在 cc<80 退化为 **FMA**（勿用）；xformers/SDPA CUTLASS 是 V100 基准（30–60 TF/s）；vLLM 官方 0.20+ 弃 V100，1Cat-vLLM(1355★) 的 `FLASH_ATTN_V100` 明确引用本仓库；HF/Dao 官方均不支持 V100。
7. 路线：A 类（文档/版本/`flash_attn.ops` shim/符号契约测试/compile-only CI/swizzle 索引恒等测试）无 GPU 可做；B 类（BMHD+接口、swizzle 重做、tile 改动）必须上 V100；C 类避免（Triton 重写、全局 use_fast_math、bf16 化、Turing 承诺、盲合 #46）。

---

## 附录 A：数据获取方法与可复现命令

```bash
# 1) 上游提交/分支/tag（GitHub API，直连可用）
curl -s 'https://api.github.com/repos/ai-bond/flash-attention-v100/commits?per_page=50'
curl -s 'https://api.github.com/repos/ai-bond/flash-attention-v100/branches?per_page=100'
curl -s 'https://api.github.com/repos/ai-bond/flash-attention-v100/tags'
curl -s 'https://api.github.com/repos/ai-bond/flash-attention-v100/commits/<sha>'   # 单提交文件清单
curl -s 'https://api.github.com/search/issues?q=repo:ai-bond/flash-attention-v100&per_page=40'  # 46 条
curl -s 'https://api.github.com/repos/ai-bond/flash-attention-v100/issues/<n>/comments?per_page=100'

# 2) fork 对比
curl -s 'https://api.github.com/repos/mejistus/flash-attention-v100/commits?per_page=20'

# 3) 本地 vs 上游文档/代码（raw.githubusercontent 直连可能超时，经代理）
https_proxy=http://127.0.0.1:7890 curl -sL https://raw.githubusercontent.com/ai-bond/flash-attention-v100/main/utils/docs/install.md
https_proxy=http://127.0.0.1:7890 curl -sL https://raw.githubusercontent.com/ai-bond/flash-attention-v100/main/flash_attn_v100/flash_attn_interface.py
diff flash_attn_v100/flash_attn_interface.py <(curl -sL .../flash_attn_interface.py)   # 实测 536 行差异

# 4) 网络备注
git ls-remote https://github.com/ai-bond/flash-attention-v100.git          # 直连 SSL_EOF 失败
https_proxy=http://127.0.0.1:7890 git ls-remote https://github.com/ai-bond/flash-attention-v100.git   # 成功

# 5) sm_70 证据链
curl -sL https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/.ci/manywheel/build_cuda.sh | sed -n '58,92p'
curl -s  https://download.pytorch.org/whl/cu126/torch/  | grep -o 'torch-2\.6\.0[^"]*x86_64\.whl' | sort -u
curl -s  https://pypi.org/pypi/torch/2.6.0/json | python3 -c 'import sys,json; print([x for x in json.load(sys.stdin)["info"]["requires_dist"] if "triton" in x or "cudnn" in x])'
curl -s  https://api.github.com/repos/pytorch/pytorch/issues/157517            # 正文含各 CUDA 版本 arch list
# dev-discuss 2.11 支持矩阵（含 12.6.3 = Maxwell…Hopper）
```

## 附录 B：主要链接索引

**上游与本地**
- https://github.com/ai-bond/flash-attention-v100（README / commits / issues / tags）
- https://github.com/ai-bond/flash-attention-v100/blob/main/utils/docs/install.md · `.../utils/docs/volta.md` · `.../utils/docs/attention.md` · `.../utils/docs/debug.md`
- https://github.com/mejistus/flash-attention-v100
- 上游重点 issue：#43 / #44 / #41 / #36 / #31 / #29 / #27 / #19 / #16 / #9 / #38 / #45

**PyTorch / sm_70**
- https://github.com/pytorch/pytorch/issues/157517（2.8–2.9 移除 Maxwell/Pascal/Volta 的 12.8/12.9 构建；12.6 保留）
- https://dev-discuss.pytorch.org/t/dropping-volta-support-from-cuda-12-8-binaries-for-release-2-11/3290
- https://github.com/pytorch/pytorch/blob/v2.6.0/.ci/manywheel/build_cuda.sh
- https://download.pytorch.org/whl/cu126/torch/ · https://pypi.org/pypi/torch/2.6.0/json
- https://pytorch.org/blog/accelerated-pytorch-2/（mem_efficient sm5x+ / FA sm80+）
- https://github.com/pytorch/pytorch/pull/91994（SDPA×FA 的历史）· https://github.com/pytorch/pytorch/issues/121558（UNFUSE_FMA）
- https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html

**xformers**
- https://github.com/facebookresearch/xformers/issues/950 · /discussions/689 · /issues/692 · /issues/813
- https://facebookresearch.github.io/xformers/components/ops.html · https://deepwiki.com/facebookresearch/xformers/2.1.2-cutlass-backend

**Triton**
- https://github.com/triton-lang/triton/pull/5066（MMAv1 弃用 → cc<80 FMA）· https://github.com/triton-lang/triton/issues/616 · https://github.com/openai/triton/issues/1279

**vLLM 生态**
- https://docs.vllm.ai/en/latest/design/attention_backends/ · https://github.com/vllm-project/vllm/pull/877 · /pull/3010 · /pull/36439 · /issues/6173 · /issues/43561 · /issues/47019
- https://github.com/1CatAI/1Cat-vLLM · https://github.com/zhinianqin/flash-attention-v100 · https://github.com/KumphanartDansiri/v100-vllm-2026

**unsloth / HF / Dao**
- https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements · https://github.com/unslothai/unsloth/issues/496 · /issues/948 · /pull/9110
- https://github.com/Dao-AILab/flash-attention · /issues/68 · /issues/1760 · /issues/543 · /issues/1225
- https://github.com/huggingface/transformers/issues/28861 · /pull/48024 · https://huggingface.co/docs/transformers/attention_interface

**其他 Volta 项目**
- https://github.com/AlexWortega/flash-attn-volta（+ RESEARCH.md / PERF.md）
- https://github.com/sirCamp/flash-attention-legacy · https://github.com/ZRayZzz/flash-attention-v100
- https://github.com/FearL0rd/ComfyUI-Flash-Attention_v100 · https://github.com/NetVoobrazhenia/ComfyUI_Flash-Attention_v100
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh（V100 数值防御实现）

**硬件/数值参考**
- NVIDIA《Programming Tensor Cores in CUDA 9》https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities （96KB smem / 64K regs）
- https://arxiv.org/html/2006.12645（Volta mma.sync.m8n8k4 布局与 fp16/fp32 累加器）· https://doi.org/10.1109/tpds.2022.3217824（Tensor Core 微基准：wmma→HMMA.884）
- https://mlinterviewnotes.com/courses/inference/03-attention-kernels/03-01-naive-attention-and-online-softmax.html（fp16 exp 溢出阈值 11.09 / 65504）

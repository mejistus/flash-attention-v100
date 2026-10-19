# 02 — flash-attention-v100 仓库全量静态审计（CUDA kernel + Python 层）

> 审计对象：`/home/eculid/Documents/Github/flash-attention-v100`
> Git 状态：branch `v100/cuda126-dropin`，HEAD = `55d685a`（2026-09-22, merge PR #45）。
> **注意**：审计时工作区存在未提交修改（`setup.py`、`pyproject.toml` 的 M 状态，见 §4），本报告以"工作区当前文件内容"为准，并在涉及处明确标注 HEAD 与工作区的差异。
> 方法：纯源码阅读 + ripgrep 模式扫描 + **用 nvcc 12.6.85 实际编译验证**（smem 常量用 `static_assert` 验证；寄存器数用等价合成 TU + `-Xptxas -v` 探测）。**无 GPU，不运行任何 CUDA 代码，无实测性能数据**。
> 标注约定：**[FACT]** = 源码/编译器可直接验证；**[HYPOTHESIS]** = 由静态证据推断、需要 GPU 或上游对照才能最终确认；**[PROBE]** = 本次审计亲手用 nvcc 编译得到的一手证据（探测文件在 `/tmp/opencode/`，未污染 repo）。

---

## 1. 架构与调用图（pybind 名称、签名、dispatch 链）

### 1.1 入口模块与符号名（[FACT]）

- `kernel/api.cpp:17-24`：`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` — 主扩展名按 build 命名为 `flash_attn_v100_cuda`（`setup.py:83`），暴露 5 个函数：`fwd`、`bwd`、`varlen_fwd`、`varlen_bwd`、`fwd_kvcache`。
- `kernel/api.cpp:26-33`：**同一 .so 内第二个 pybind 模块** `flash_attn_2_cuda`，函数名相同 → 使 `import flash_attn_2_cuda` 这一上游命名习惯可用；`setup.py:166-177` 在 build 后创建 `flash_attn_2_cuda.so -> flash_attn_v100_cuda*.so` 符号链接。
- `kernel/api.cpp:308-357`：TorchBind 命名空间 `flash_attn_v100`，5 个 schema（`torch.ops.flash_attn_v100.*`），与 pybind 参数一一对应：
  - `fwd(Tensor(a!) q, Tensor k, Tensor v, Tensor? out, Tensor? alibi_slopes, float p_dropout, float softmax_scale, bool is_causal, int window_left, int window_right, float softcap, bool return_softmax, Generator? gen) -> Tensor[]`（:309-315）
  - `bwd(Tensor dout, Tensor q, Tensor k, Tensor v, Tensor out, Tensor softmax_lse, Tensor? dq, Tensor? dk, Tensor? dv, Tensor? alibi_slopes, float p_dropout, float softmax_scale, bool is_causal, int window_left, int window_right, float softcap, bool deterministic, Generator? gen, Tensor? rng_state) -> Tensor[]`（:317-324）
  - `varlen_fwd(..., Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k, Tensor? block_table, ..., bool zero_tensors, ..., int num_splits) -> Tensor[]`（:326-335）
  - `varlen_bwd(..., bool zero_tensors, ..., Tensor? rng_state) -> Tensor[]`（:337-346）
  - `fwd_kvcache(Tensor(a!) q, Tensor kcache, Tensor vcache, Tensor? k, Tensor? v, Tensor? seqlens_k, Tensor? rotary_cos, Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? leftpad_k, Tensor? block_table, Tensor? alibi_slopes, Tensor? out, float softmax_scale, bool is_causal, int window_left, int window_right, float softcap, bool is_rotary_interleaved, int num_splits) -> Tensor[]`（:348-357）

### 1.2 调用链（[FACT]）

```
python: flash_attn_func / flash_attn_varlen_func / flash_attn_with_kvcache        (flash_attn_v100/flash_attn_interface.py:147/315/366)
  └─ torch.autograd.Function  →  C++ pybind 函数 (kernel/api.cpp → kernel/*.cu flash_attention_* wrapper)
       └─ dispatch_attention_features(is_causal, is_alibi, is_softcap, is_window, is_dropout, is_paged, is_rope, is_interleaved,
                                      lambda<…compile-time bools…>)                   (include/template.h:266-290)
            └─ launcher_flash_attention_*<D>  (dense fwd: kernel/forward.cu:225; dense bwd: kernel/backward.cu:5xx)
                 └─ kernel<<<grid, block, smem, stream>>>   — 唯一 extern smem = KernelConfig<D>::TOTAL_SMEM
```

- 编译期模板参数顺序（5 个 kernel 均以 512 线程、`__launch_bounds__(THREADS_PER_BLOCK, 2)` 启动）：
  - dense forward：`flash_attention_forward_kernel<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT>`（kernel/forward.cu:25-26）
  - varlen forward：`+ IS_PAGED`（kernel/forward_varlen.cu:25-26）
  - dense/varlen backward：`<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_DROPOUT>`（kernel/backward.cu:25-27、backward_varlen.cu:25-27）
  - kv-cache：`<D, IS_CAUSAL, IS_ALIBI, IS_SOFTCAP, IS_WINDOW, IS_PAGED, IS_ROPE, IS_INTERLEAVED>`（kernel/forward_kvcache.cu:25-26，无 dropout）
- `dispatch_attention_features` 的关键语义（include/template.h:277-284）：`chk_softcap = is_softcap && !is_dropout`、`chk_dropout = is_dropout && !is_softcap` — **softcap 与 dropout 互斥**（wrapper 层也有对应 `TORCH_CHECK`，如 kernel/forward.cu:343）。

### 1.3 Grid/Block 布局（[FACT]）

| Kernel | grid | block | 证据 |
|---|---|---|---|
| dense fwd | `(ceil(M/BM), 1, B*H_Q)` | 512 | kernel/forward.cu:256-257 |
| varlen fwd | `(ceil(max_seqlen_q/BM), 1, B*H_Q)`，batch 由 device 端解析 | 512 | kernel/forward_varlen.cu:318-319, BlockInfo.init_q（template.h:55-68） |
| dense bwd | `(max(grid_dq, grid_dkv), 2, B*H_Q)`；`blockIdx.y=0`→dQ 相（backward.cu:59），`y=1`→dKV 相（:257） | 512 | kernel/backward.cu:527-532 |
| varlen bwd | `(max(grid_dq, grid_dkv), 2, B*H_Q)`；y=0→dQ（:63），y=1→dKV（:292） | 512 | kernel/backward_varlen.cu:584-586 |
| kv-cache | `(ceil(T_Q/BM), 1, B*H_Q)` | 512 | kernel/forward_kvcache.cu:342-343 |

- dKV 相的 GQA 处理：dense dKV 用 `blockIdx.z` 覆盖 `B*H_K`，`bthd_idx >= B*H_K` 直接 return（backward.cu:269-272），每块内再循环 `H_Q/H_K` 个 q-head 组累加同一 KV head 的梯度（backward.cu:350-358）；q→kv head 映射 `head_idx / (H_Q/H_K)`（template.h:58、140；forward_kvcache.cu:81）与 dKV 的 `(bthd_idx % H_K)*(H_Q/H_K)+group` 等价（整除关系一致）。GQA 时 dense dKV 的 grid.z 是 `B*H_Q`，其中 `B*(H_Q-H_K)` 个块直接空转（浪费，见 §7）。
- varlen bwd 的 GQA：dKV 相把 dK/dV 展开到 `[T_K, H_Q, D]`（backward_varlen.cu:762-763），kernel 后再用 `at::sum_out` reshape 归约（backward_varlen.cu:800-801）。

### 1.4 BlockInfo（include/template.h:15-243）

- `init_q`（Q-centric：fwd/dQ）：解析 dense/varlen 元数据，计算 `seqlen_offset = seqlen_k - seqlen_q`，并按 causal/window 裁剪 KV 迭代区间 `[block_min, block_max)`（template.h:98-111）。
- `init_kv`（KV-centric：dKV）：反向裁剪 Q 迭代区间（template.h:180-193）。
- 偏移计算：`q_offset/kv_offset/lse_offset/dmask_offset`（template.h:199-242）；varlen 下 q/k/v 是 ragged 布局 `[T, H, D]`，dense 下 `[B, H, S, D]`。
- `SEQUSED_K` 支持：`seqlen_k = min(seqlen_k, used_k)`（template.h:65-68），用于 varlen fwd/bwd（`seqused_k` 参数）。

---

## 2. 功能矩阵（implemented vs accepted-but-ignored）

### 2.1 数据类型 / 尺寸（[FACT]）

| 项 | 状态 | 证据 |
|---|---|---|
| dtype | **仅 fp16**；q/k/v/out 全部 `TORCH_CHECK(... == kFloat16)` | forward.cu:321、backward.cu:606-609、forward_varlen.cu:400、backward_varlen.cu:（同模式）、forward_kvcache.cu:443-444 |
| 累加精度 | fp32（mma `...f32.f16.f16.f32`；S/O/dQ/dK/dV/lse 全 fp32 smem/global） | mma_m8n8k4.h:34、kernel.h:10-18；forward.h:45-58 |
| 支持的 head dim | kernel：D=16/32/64/128/256（`D<=256 && D%8==0`） | forward.cu:334-335；switch 只有 5 档（forward.cu:410-417） |
| Python 层 head dim | 任意 8 的倍数 ≤256 → **向上取整 padding 到 16/32/64/128/256** 再切回；scale 用原始 head_size | flash_attn_v100/flash_attn_interface.py:23-32、73-85 |
| bf16/fp32/fp8 | 无 | §3 grep（`bf16/bfloat16` 零命中）；无 descale 参数 |

### 2.2 Attention 功能（[FACT]，除注明外）

| 功能 | 状态 | 证据 / 备注 |
|---|---|---|
| causal | 实现（mask 条件 `(global_n - seqlen_offset) > global_m` → NEG_INF） | mat_mul.h:92、103 |
| sliding window | 实现，`window_left/window_right`，`-1` 关闭；符号语义 `key < q-WIN_L` / `key > q+WIN_R` 屏蔽 | mat_mul.h:106、136；BlockInfo 裁剪 template.h:102-111、184-193 |
| softcap | 实现 `SOFTCAP*tanh(val/SOFTCAP)`；与 dropout 互斥 | mat_mul.h:115-117；template.h:280-281 |
| alibi | 实现，`slopes` 支持 `[H]` 或 `[B,H]`；bias `-slope*|m-(n-offset)|` | forward.cu:353-360；mat_mul.h:113 |
| dropout | 实现，自研 Philox4x32-10（seed+offset），mask 存 ±1 half；`p∈[0,1)` | include/philox.h；softmax.h:106-138 |
| return_softmax / dmask | 仅 dense fwd 与 varlen fwd 实现（kvcache 无）；dmask layout dense `[B,H,M,N]` / varlen `[T_Q,H_Q,max_seqlen_k]` | forward.cu:398-402；forward_varlen.cu:527-531；softmax.h:129-137 |
| deterministic | **明确拒绝**：`TORCH_CHECK(!deterministic, ...)`；Python 层 warning 后强转 False | backward.cu:602；backward_varlen.cu:665；flash_attn_v100/flash_attn_interface.py:161-163、334-336 |
| split-KV (`num_splits`) | **未实现**：仅允许 0/1 | forward_varlen.cu:419；forward_kvcache.cu:460 |
| rotary (RoPE) | 仅 kvcache 路径；interleaved（默认 True）与 half-split 两种；`rotary_dim = cos.size(1)*2`，须 `%16==0` 且 ≤D；cos/sin fp16 `[seqlen_ro, rotary_dim/2]` | include/rotary.h；接口参数 flash_attn_v100/flash_attn_interface.py:372-373、428 |
| GQA / MQA | 实现（`H_Q % H_K == 0`）；varlen bwd 通过 expand+sum 归约 | template.h:58；backward_varlen.cu:762-763、800-801 |
| varlen | 实现（fwd+bwd，ragged `[T,H,D]`，device 端 batch 解析） | template.h:55-68；forward_varlen.cu / backward_varlen.cu |
| paged KV | 实现（fwd varlen + kvcache），**但读路径有页边界缺陷**，见 §6 P1-3 | forward_kvcache.cu:199-208；forward_varlen.cu:182-185 |
| `leftpad_k`（varlen fwd） | **accepted-but-ignored**：host 端校验并拒绝与 paged 同用，但 device kernel 从不读它 | forward_varlen.cu:37（仅形参）、466-475（校验）；kernel body 无引用 |
| `leftpad_k`（kvcache） | 实现（更新、读、rope 位置都含 leftpad） | forward_kvcache.cu:84、200-211；rotary.h:62、177 |
| `seqused_k` | 实现（truncate 有效 KV 长度） | template.h:65-68、146-149 |
| `cache_batch_idx` | 实现（非 paged），paged 时禁止 | forward_kvcache.cu:83、478 |
| KV-cache 原地扩展 | 实现：`k/v` 输入被 RoPE 后写入 `k_cache/v_cache`（每个 Q-block 每个 head 重复写，见 §6 P2-2） | forward_kvcache.cu:134-140；rotary.h:14-149 |
| `cache_seqlens` 更新 | **不做**（上游 `flash_attn_with_kvcache` 会原地更新 `cache_seqlens`） | flash_attn_v100/flash_attn_interface.py:366-434（函数体无写回） |
| causal 自动关闭 | M==1 / max_seqlen_q==1 / T_Q==1 且无 alibi 时 `is_causal=false`（decode 优化） | forward.cu:339；forward_varlen.cu:422；forward_kvcache.cu:463 |
| kvcache 的 window 修正 | causal 时强制 `window_right=0`；softcap 与 window/alibi 互斥 | forward_kvcache.cu:464、467-469 |

### 2.3 Python API 面（drop-in 对照）（[FACT]）

- 包结构：`flash_attn/`（shim：`__init__.py`、`flash_attn_interface.py` 25 行纯 re-export、`bert_padding.py` 146 行）+ `flash_attn_v100/`（真实现）+ `flash_attn_v100_cuda` 扩展。
- `flash_attn/__init__.py:19`：`__version__ = "2.8.3"`（伪造上游版本号）；`setup.py:118-124` 还会伪造 `flash_attn-2.8.3.dist-info/METADATA` 以便 `pip`/importlib 元数据检查通过。
- 函数面齐全：`flash_attn_func`、`flash_attn_varlen_func`、`flash_attn_with_kvcache`、`flash_attn_qkvpacked_func`、`flash_attn_kvpacked_func`、`flash_attn_varlen_qkvpacked_func`、`flash_attn_varlen_kvpacked_func`，以及 `*_gpu` 别名（flash_attn_v100/flash_attn_interface.py:444-548）。packed 版本是 `unbind` + 调 `flash_attn_func` 的薄封装（梯度经视图回填）。
- 上游缺口（HYPOTHESIS：drop-in 兼容性缺口清单，详见 `research/03_api_parity.md`）：无 `flash_attn.ops.*` / `flash_attn.modules.*` / `flash_attn.layers.*` 子包；无 fp8 的 `q_descale/k_descale/v_descale` 参数；kvcache 不接受 `s_aux`/MTP 类新参。
- `flash_attn_varlen_func` 的 Python 签名含 `block_table`（paged）但**不含** `seqused_k`/`leftpad_k`（接口层未暴露，C++ 层支持）— flash_attn_v100/flash_attn_interface.py:315-332。

### 2.4 仓库自带测试覆盖（[FACT]）

- `test.py:114` `test_combined()`：**仅 dense fwd+bwd**，与 torch 参考对比；覆盖 D=16/32/64/128/256、序列 16~8192、causal 开/关。
- `test.py:144-145`：`if causal and M > N: continue` — **恰好跳过**了 §6 P1-2 的 fully-masked-row 场景。
- 无 dropout / alibi / window / softcap / varlen / kvcache / GQA（H 只有 1/16/32 且 kv heads = heads）的任何测试。

---

## 3. sm_70 合规性 grep（固定清单，全部 `kernel/ include/ flash_attn*/`）

**[FACT] 以下模式命中数全部为 0**（`grep -rIn`，本次审计复跑）：

`ldmatrix`、`cp.async`（含 `cp.async.bulk`/TMA）、`m16n8k8`、`m16n8k16`、`bfloat16`、`bf16`、`wgmma`、`stmatrix`、`movmatrix`、`redux`、`__reduce_`、`elect.sync`、`cuda::pipeline`、`cuda::barrier`、`mbarrier`、`cluster`、`__grid_constant__`、`cudaMemAdvise`、`L2::`、`createpolicy`、`cuda::memcpy_async`、`dp4a`、`__nv_*`、`cuda::std`、`CUDART_VERSION`、`__builtin_*`。

**[FACT] `__CUDA_ARCH__` 全仓库仅 2 处**，且都是显式拒绝非 sm_70 编译：

- `include/mma_m8n8k4.h:37`：`#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 700)` → `#error "Volta HMMA m8n8k4: This header is for sm_70 ONLY! ..."`
- `include/mma_m16n16k16.h:20`：同样的 `#error "Volta WMMA: This header is for sm_70 ONLY!"`

**[FACT] MMA 实现**：

- 默认后端：`include/mat_mul.h:9-12` 选择顺序 `#ifdef MMA_NATIVE → <mma.h>`（nvcc 的 `nvcuda::wmma`），`#elif MMA_884 → include/mma_m8n8k4.h`，`#else → include/mma_m16n16k16.h`（仓库自研 wrapper）。
- `mma_m8n8k4.h:475-500` 发出真正的 PTX `mma.sync.aligned.m8n8k4.{row,col}.{row,col}.f32.f16.f16.f32`（sm_70 合法，HMMA.884）。
- `mma_m16n16k16.h` 是自研 m16n16k16 wrapper（HMMA.884 组合实现 + 自带 swizzle；文件头注释 :1-18）。
- 结论：**源码级完全合规 sm_70；且通过 `#error` 主动杜绝误编到 sm_75+**（Triton/Blackwell 类新特性完全不存在）。`setup.py:54` 只发 `-gencode arch=compute_70,code=sm_70`，无 PTX 前向兼容产物（设计目标如此）。

---

## 4. CUDA 12.6 兼容性风险 & "为什么 setup.py 声称 CUDA>=12.9"

### 4.1 事实链

- **[FACT] 源码层面不存在任何 12.9 依赖**：§3 的扫描（TMA/cp.async/head-dim 新 API/`cuda::std`/`CUDART_VERSION` 等全 0）+ 全部 include 只有 `<cuda.h>/<cuda_runtime.h>/<cuda_fp16.h>`（以及 `MMA_NATIVE` 时的 `<mma.h>`）。编译产物仅用到 PTX `mma.sync m8n8k4`、`wmma` 相关和 `__shfl_*`、`ld/st.shared.v4` 等 Volta 特性。
- **[FACT] ">=12.9" 的来源是环境策略而非代码依赖**：
  - HEAD 版 `setup.py` 要求 `torch.cuda.is_available()` 且 `torch.version.cuda >= 12.9`（工作区 diff 可见，见 4.2），`pyproject.toml` HEAD 版把 `torch==2.10.0+cu129` 直接钉死（wheel URL）。
  - `utils/docs/install.md:10,13`（安装 cuda_12.9.1）+ `:41-45`（理由：NVIDIA 从 CUDA 13 起弃 Volta；PyTorch 从 CUDA 12.8/12.9 构建起删去 Volta 支持 → "torch 2.10.0+cu129 是最后一个带 Volta 的栈"）。
  - 即：[HYPOTHESIS] 12.9 是作者验证环境的"最后一个带 Volta 的官方栈"选择，**不是 kernel 源码要求**；把它理解为 API 下限会误伤 CUDA 12.6 目标。
- **[FACT] 工作区（未提交）已经把闸门改为 12.6 默认**：
  - `setup.py:139-141`：`min_cuda = os.environ.get("FLASH_ATTN_MIN_CUDA", "12.6")`，注释明言"12.6 是我们验证过的最老的完整支持 Volta 的工具链"；:148-152 版本不满足才 raise；:153-159 无 GPU 只告警不报错；错误提示文本改为 `torch==2.6.0 --index-url .../cu126`（:47、:132）。
  - `pyproject.toml` 工作区版：`torch` 不再钉轮子、`numpy>=2.0.0`。
  - **[FACT] 因此 HEAD 版在 torch 2.6.0+cu126 下会直接 raise（"12.6 < 12.9"），必须依赖这组未提交修改或 `FLASH_ATTN_MIN_CUDA=12.6`**。这是本仓库向"CUDA 12.6 + torch 2.6.0 drop-in"迁移的关键前提。

### 4.2 CUDA 12.6 实际编译验证（[PROBE]）

- 用 `nvcc 12.6.85`（`/mnt/envs/v100/bin/nvcc`）+ `-gencode arch=compute_70,code=sm_70 -std=c++17` 成功编译了包含本仓库 `include/*.h`（含两个 MMA wrapper、swizzle、philox）的验证 TU（smem 常量 `static_assert` 全部通过，见 §5）。→ 12.6 工具链可编译本仓库全部头文件与 PTX。
- 编译参数兼容（`setup.py:51-63`）：`-O3/-std=c++17/-gencode/-U__CUDA_NO_HALF*__/--expt-relaxed-constexpr/--expt-extended-lambda/--use_fast_math/-Wno-deprecated-gpu-targets/--threads` 在 12.6 全部存在可用（与 `research/04_build_audit.md` 独立结论一致）。
- **主机编译器风险（本机实测）**：系统 `gcc 16.2.1` 与 nvcc 12.6 不兼容：`host_config.h:143: #error -- unsupported GNU version! gcc versions later than 13`；加 `-allow-unsupported-compiler` 后仍然在 libstdc++(gcc16) 头文件层炸（`__is_array` / `user-defined literal operator`，最小 TU 也无法编译）。用 conda 的 `x86_64-conda-linux-gnu-g++ 12.4` + `-ccbin` 编译成功。→ 构建配方必须给 nvcc 指定 gcc≤13（torch 2.6.0 的 cpp_extension 会自动从环境变量 `CC` 生成 `-ccbin`，细节见 04_build_audit）。
- **[FACT] 泛型风险**：`sm_70` 在 12.6 里已属 deprecated 目标（用 `-Wno-deprecated-gpu-targets` 静音），CUDA 13 起删除 — 与本项目"只支持 V100"定位一致，不构成 drop-in 阻塞。

---

## 5. 每 kernel 的 smem / 寄存器 / occupancy 分析

### 5.1 Smem 布局与 TOTAL_SMEM（[PROBE]：用 nvcc 对真实头文件 `static_assert` 验证）

布局定义：forward `include/forward.h:42-61`（union `fdo`：q / reuse_kv{k|v} / reuse_sp{s|p} / o + 尾部 row_max/row_sum）；backward `include/backward.h:64-104`（两相 union：bdq{reuse_kv, dO, q, s, reuse_sdOVS{dOV|dS}, dQ} 与 bdkv{k, v, reuse_qdO{dO|q}, reuse_sp{s|p}, reuse_dOVS{dOV|dS}, dK, dV} + 尾部 lse/row_dot）。`TOTAL_SMEM = round_up(sizeof(SmemLayout),128)`（forward.h:61 / backward.h:104）。

**编译验证过的精确值**（单位 bytes；探测 TU：`/tmp/opencode/smem_check_{fwd,bwd}.cu`，`static_assert` + 运行时 printf 双确认）：

| D | fwd BM/BN | fwd TOTAL_SMEM | fwd KB | bwd TOTAL_SMEM | bwd KB | 每 SM 可容纳 CTA（98304B） |
|---|---|---|---|---|---|---|
| 16 | 16/512 | **50816** | 49.6 | **72448** | 70.8 | 1 / 1 |
| 32 | 32/256 | **55552** | 54.3 | **75264** | 73.5 | 1 / 1 |
| 64 | 64/128 | **74240** | 72.5 | **84992** | 83.0 | 1 / 1 |
| 128 | 32/176 | **92416** | 90.3 | **91136** | 89.0 | 1 / 1 |
| 256 | 32/64 | **90368** | 88.3 | **90624** | 88.5 | 1 / 1 |

- 分解示例（D=128 fwd：BM=32/BN=176/D_STRIDE=128/N_STRIDE=176）：q=8KB + reuse_kv=44KB + reuse_sp(s/p 同为 22KB) + o=16KB + row_max/row_sum=256B = 92416B ✔；bwd D=128 dQ 相 90112B（reuse_kv 28KB + dO/q 各 8KB + s 14KB + dOV/dS 14KB + dQ 16KB）+ lse/row_dot 1KB = 91136B ✔。
- 设计含义（[FACT]）：**所有配置 1 CTA/SM**（最小配置 50816B×2 > 98304B），smem 是硬性 occupancy 上限；512 线程 = 16 warps = Volta 64-warp 容量的 25%。
- **[FACT] >48KB 的 opt-in 都做了**：5 个 launcher 都调用 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)`（forward.cu:283、forward_varlen.cu:351、backward.cu:561、backward_varlen.cu:618、forward_kvcache.cu:394），并有 `TORCH_CHECK(smem <= MAX_SMEM_PER_SM)`（=98304，kernel.h:28；forward.cu:248、backward.cu:519）。若漏掉任一处，运行时会直接 launch 失败 — 目前 5 处齐全。
- smem 初始化：`WMMA_GEMM_INIT_SMEM` 对整个 layout 做 16B 粒度的 v4 清零（gemm_smem.h:44-68），forward 先整体清零再单独填 NEG_INF 到 row_max（forward.cu:125-128）；backward 在每相入口再次整体清零（backward.cu:317 附近、:140）。

### 5.2 `__launch_bounds__(512, 2)` 的寄存器代价（[PROBE]：合成热循环 + `-Xptxas -v`）

所有 5 个 kernel 都声明 `__launch_bounds__(THREADS_PER_BLOCK=512, 2)`（forward.cu:26、backward.cu:27、forward_varlen.cu:26、backward_varlen.cu:27、forward_kvcache.cu:25）。其第二参数要求"每 SM 至少 2 个 block"，即把每线程寄存器压到 `65536/(512*2)=64`。但 §5.1 已证明 **2 CTA/SM 在 smem 上永远不可能** → 该上限纯属浪费。

为量化代价，审计构造了两个"等价热循环"合成 TU（复制真实调用序列：LOAD_TILE→SCORES→SOFTMAX→GRADIENTS；及 backward dQ 相 LOAD(dual)→DOT_PRODUCT→SCORES→SOFTMAX_GRADIENT→dGRADIENTS），用真实 `include/*.h` 以 `-O3 --use_fast_math -arch=sm_70 -Xptxas -v` 编译（探测文件 `/tmp/opencode/synth_{fwd,bwd}.cu`）。**[HYPOTHESIS] 合成 TU ≠ 完整 kernel**（缺少完整外层循环/多 block 展开，寄存器占用可能被低估），但趋势明确：

| 合成 kernel | `-Xptxas -v` 寄存器 | spill |
|---|---|---|
| fwd D=128，launch_bounds(512,**1**) | **126** | 0 |
| fwd D=128，launch_bounds(512,**2**) | **64** | 8B stack / 8B st / 12B ld spill |
| fwd D=128 + dropout， (512,2) | 64 | 8B stack / 8B / 12B |
| fwd D=64， (512,2) | 64 | 0 |
| fwd D=16， (512,2) | 64 | 8B stack / 4B / 4B |
| bwd dQ D=128， (512,**1**) | **111** | 0 |
| bwd dQ D=128， (512,2) | 64 | 0（本子集） |
| bwd dQ D=64， (512,2) | 64 | 8B / 4B / 4B |
| bwd dQ D=16， (512,1) | 64 | 0 |

结论（[HYPOTHESIS]→建议 GPU 验证）：自然寄存器需求 ~111-126 regs/thread；`(…,2)` 把值硬压到 64 并引入少量 spill，同时寄存器文件只用掉 65536 的 50%（64×512/65536）。由于 smem 决定 1 CTA/SM，**把上限放宽到 1（或去掉第二参数）几乎零风险且很可能提速**（保留 512 的单 CTA 资源上限 128 regs）。见 §7 优化 #1。

（本机无法编译完整扩展拿真实 ptxas 数据：`/mnt/envs/v100` 里 torch 尚未装好，`import torch` 仍失败；完整 kernel 的寄存器数据列入 §8。）

---

## 6. 正确性风险清单（P0/P1/P2 + file:line + 最小修复草案）

### P0

**P0-1 varlen + dropout 的前/后向 RNG 索引 stride 不一致 → 反向静默算错梯度**
- 机制（[FACT]）：dropout 的随机数是按元素全局坐标哈希出来的：`addr_plx = (GLOBAL_ROW_OFFSET+row)*GLOBAL_N + (GLOBAL_COL_OFFSET+col)`，`philox(seed, offset + (addr_plx>>2))`（forward：include/softmax.h:109-111、尾段 :154-156；backward：include/softmax.h:290-291）。
- 调用参数（[FACT]）：
  - varlen fwd 传 `GLOBAL_N = max_seqlen_k`：`kernel/forward_varlen.cu:232-233`（`block.q_base+block.start_q, start_kv, max_seqlen_k, ..., H_Q*max_seqlen_k`）。
  - varlen bwd 传 `GLOBAL_N = block.seqlen_k`（每 batch 的真实 KV 长度）：dQ 相 `kernel/backward_varlen.cu:259`、dKV 相 `kernel/backward_varlen.cu:490`。
- 后果：只要某个 batch 的 `seqlen_k != max_seqlen_k`（varlen 的日常情形），**fwd 用一套 dropout mask、bwd 重算出另一套**，梯度与 loss 不一致，且不报错。
- 对照（[FACT]）：dense 路径一致（fwd `kernel/forward.cu:181-182` 与 bwd `kernel/backward.cu:229`、:438 都用 `N`）。
- 最小修复草案：统一为同一个 stride（例如两者都用 `max_seqlen_k`，backward 的 `GLOBAL_N` 改为拉取同一值；或统一改写为 `philox % (q_pos * N_global + kv_pos)` 且 fwd/bwd 都传同一表达式）。修复后需要 GPU 上跑 dropout>0 的 varlen fwd/bwd 与 torch 参考对比（当前 test.py 无此覆盖）。
- 附带（P2 级设计问题，见 P2-5）：该 RNG 索引不含 head 维 → 所有 head（以及 dense 的所有 batch）共用同一 mask，与上游 FA2（per batch-head 不同随机流）行为不同。

### P1

**P1-1 dense backward 返回未初始化的 `dsoftmax_sum`**
- [FACT] `kernel/backward.cu:691`：`dsoftmax_sum = torch::empty({B,H_Q,M}, fp32)`；:719 原样返回；只有 `M==0||N==0` 早退路径才会 `zero_()`（:699）。整个 kernel 无任何写它的参数（backward.cu 的 launcher 参数列表 :563-569 无 softmax_d）。
- 对照 [FACT]：varlen 后向**会**写 softmax_d（`kernel/backward_varlen.cu:105-115` 早退清零、`:144`、`:277` 由 row_dot 写入）→ 明显是 dense 侧遗漏。
- 最小修复草案：像 varlen 一样在 dQ 相把 `row_dot` 写到 `dsoftmax_sum`（最简单是 dense bwd 增加一行输出拷贝，成本≈0 新计算），或至少 `zero_()`。

**P1-2 fully-masked 行：P 全 1（输出为 mean(V)）而不是 0**
- 机制（[FACT]）：`include/softmax.h:92-104`：整行全被 mask 时 `thread_max = NEG_INF`、`new_max = NEG_INF`，`exp_diff = exp(NEG_INF-NEG_INF) = exp(0) = 1`，每元素 `exp(fmax(sS-new_max,-80)) = exp(0) = 1` → 该行 P 全 1，`row_sum = 有效列数`，`Out row = (ΣV)/count`（垃圾值）；lse 写为 `row_max+log(max(sum,1e-24)) ≈ -1e30`（forward.cu:216-219）。
- 可达性（[FACT]+[HYPOTHESIS]）：dense/varlen causal 且 `M>N`（`seqlen_offset<0`）时，Q block 里 `m < -offset` 的行必然全 mask；只要该 Q block 的 `block_max>0`（块内较后的行有有效 key）循环就会执行（BlockInfo 裁剪 template.h:98-111 只跳过"整块无有效 key"的块）→ 一定能触发。且 **repo 自带测试恰好跳过该组合**（test.py:144-145 `if causal and M > N: continue`）。
- 上游语义：应为 Out=0、lse=-inf。最小修复草案：在 softmax.h 收尾处判 `thread_max==NEG_INF`（或用 row_sum==0 判）→ 该行 P 写 0、lse 写 -inf；或在内核 epilogue 里对 `sRowSum[row]==0` 的行直接清 Out。同样需要 GPU 验证。

**P1-3 paged KV 读路径假设"一个 tile 不跨页"，256 的检查不够**
- [FACT] 读 tile 只算一次页号：`forward_kvcache.cu:199-208`（`page_idx=(start_kv+leftpad)/block_page; ... K_cache + phys_page*kv_block_stride + page_offset*stride_k_s`），varlen 同构 `forward_varlen.cu:182-185`；随后 `WMMA_GEMM_LOAD_TILE` 以 `stride_k_s` 连续读 `BLOCK_N` 行。
- 门槛检查只有 `page_block_size % 256 == 0`（forward_varlen.cu:436、forward_kvcache.cu:486）。跨页发生的充分必要条件：存在 tile 起点 `start_kv=k*BLOCK_N` 使 `start_kv % page + BLOCK_N > page`。对常见页大小：
  - D=16（BN=512）：page=256 时必然跨页（512>256）；page=512 及以上才安全。
  - D=128（BN=176）：page=256/512/1024 全部存在跨页 tile（176 不整除任何 2 的幂）。
  - D=32（BN=256）/D=64（BN=128）：page 为 256 的倍数时安全。
- 后果：跨页 tile 的 K/V 读错页 → 静默错误结果（若 block_table 映射恰好物理连续则碰巧正确）[HYPOTHESIS 影响面]。
- 对照：**写路径是安全的**（逐 token 页号，rotary.h:64-76）。
- 最小修复草案：① 把检查收紧为 `page_block_size % BLOCK_N == 0`（D 相关），或 ② 读路径像写路径一样逐行查页（性能损失小：每 tile 每行一次除法），推荐 ② 以便支持任意 256 倍数的页大小。

**P1-4 kvcache 非 causal 时 Q 的 RoPE 位置对 T_Q>1 全行共用同一位置**
- [FACT] `include/rotary.h:200-202`：`q_rope_stride = (IS_CAUSAL || IS_WINDOW) ? 1 : 0; pos = rope_base + row * q_rope_stride`；`rope_base = cache_seqlen + leftpad (+ start_q if causal/window)`（rotary.h:177）。
- 即 causal/window 开启时每行 pos 正确（`cache_seqlen+leftpad+start_q+row`，与其被写入 cache 的位置一致，rotary.h:62）；**非 causal（kvcache 默认！）时所有行都取 `pos=cache_seqlen+leftpad`** —— 仅 T_Q==1 的 decode 特例正确（而 T_Q==1 时 causal 又会被 wrapper 自动关掉，forward_kvcache.cu:463，放大触发概率：**默认即踩**）。
- 触发：任意 `flash_attn_with_kvcache(..., rotary_cos/sin, causal=False)` 且 T_Q>1（带 cache 的 prefill）。
- 最小修复草案：去掉 stride 条件，始终 `pos = cache_seqlen + leftpad + (start_q if causal/window else 0) + row`（即非 causal 也用 1）。

### P2

**P2-1 kvcache 早退发生在 KV-cache 更新之前**：[FACT] `forward_kvcache.cu:108-119`（`block.block_min>=block.block_max || total_seqlen_k==0` 即写 0 返回）在读访问之前；而缓存更新在 `:134`。若该分支被触发，本次的 `k/v` 新 token 不会进 cache → 后续调用缺 token。[HYPOTHESIS] 常见参数下难以触发（causal 时 offset≥0，块内总有 key），但语义脆弱；修复：把 `WMMA_GEMM_UPDATE_KVCACHE` 挪到早退判定之前。

**P2-2 KV-cache 更新被每个 Q-block × 每个 q-head 重复执行**：[FACT] 更新的调用在 kernel 顶部（forward_kvcache.cu:134-140），grid 为 `(ceil(T_Q/BM),1,B*H_Q)` → 相同 `(batch, kv_head)` 的 cache 行被 `ceil(T_Q/BM) × (H_Q/H_K)` 次重复写同值（并发写同值=良性竞态，当前语义下 idempotent；一旦 RoPE/写入语义变化就会变危险）。性能上也是成倍的 global 写放大（见 §7 #2）。

**P2-3 varlen fwd 的 `leftpad_k` accepted-but-ignored**：[FACT] `kernel/forward_varlen.cu:37` 形参存在且 host 校验（:466-475、拒绝与 paged 同用），但 kernel body 零引用 → 用户传了不生效、也不报错。

**P2-4 `-DMMA_884` 构建下 causal/window mask 的 lane 映射与 fragment 布局不匹配**：[FACT] `include/mat_mul.h:54-55` 的 `row_causal/col_causal` 常量 + :86-87 的元素索引分解是按 **m16n16k16** accumulator 布局写的（与默认 `mma_m16n16k16.h` wrapper 配套）；而 `include/mma_m8n8k4.h:270-282` 明确其 8 元素 fragment 布局为 `r0=(lid&1)+((lid>>4)&1)*4, c0=lid&2`（store 实现 :388-430）。MMA_884 下线号公式多了 `((lid>>2)&1)*8`（行）/`((lid>>3)&1)*8`（列）两项 → 行/列定位错位，mask 会应用到错误的逻辑元素。[HYPOTHESIS]（需要 GPU 或逐 lane 仿真确认）— 只影响显式 `MMA_884=1` 的可选构建，默认构建不受影响。

**P2-5 dense/dropout 的随机 mask 与 head/batch 无关**：[FACT] RNG 索引（softmax.h:109-111,290-291）与写出位置（forward.cu:179-182 的 `dmask_ptr` 已含 head 偏移，但数值相同）都不含 head/batch → 同一 (m,n) 在所有 head、所有 batch 上 dropout 图案完全一致；上游 FA2 的 rng 流是按 (batch, head) 区分的。影响：统计独立性与精度（非功能性崩溃），以及 `return_softmax` 的 dmask 在某些 head 上"看起来"正确。

**P2-6 `return_softmax` 的 dmask 语义**：[FACT] dmask 写的是 ±1 标志（保留=0x3C00=+1.0，丢弃=0xBC00=-1.0；softmax.h:130-136、:164-167），不是概率值。[HYPOTHESIS] 与上游 FA2 `S_dmask` 的语义（存 dropout 后的概率）不同 → `return_attn_probs` 消费者若直接用 dmask 会有差异；需要对照上游实现确认。

**P2-7 `num_splits>1` 不支持**：[FACT] forward_varlen.cu:419、forward_kvcache.cu:460（"not supported now, for future develop"）。decode（T_Q 很小、KV 很长）场景无 split-KV 并行，属性能/覆盖缺口而非正确性 bug。

**P2-8 `.cpu()` 同步**：backward 拿到 `rng_state` 后 `.cpu()` 读取（backward.cu:661-664、backward_varlen.cu:733-735）→ 每步 dropout 训练会引入一次 device→host 同步（微性能问题，非 bug）。

---

## 7. 优化机会排名（file:line / 预期收益 / 风险 / 是否可立即改）

1. **放宽 `__launch_bounds__` 第二参数**（5 kernel 各处：forward.cu:26、backward.cu:27、forward_varlen.cu:26、backward_varlen.cu:27、forward_kvcache.cu:25）。收益：解除 64-reg 硬压缩（自然需求 111-126 regs，§5.2），消除 D=16/64/128 的少量 spill，提升 ILP；这是"1 CTA/SM 已由 smem 锁死"前提下白捡的资源（512 线程单 CTA 上限 128 regs）。风险：低（仍受 smem 限制，occupancy 不变；寄存器上限变化只影响编译分配）。**可立即改，但收益幅度需 GPU 实测**。
2. **KV-cache 更新只做一次**（forward_kvcache.cu:134-140）。用谓词限制为"第一个 Q-block 且拥有该 kv_head 的第一个 q-head"（例如 `block_idx==0 && head_idx % (H_Q/H_K)==0`）即可把写放大从 `ceil(T_Q/BM)*H_Q/H_K` 降到 1；写同值竞态也随之消失。风险：低-中（需保证所有 token 都覆盖；T_Q 超过 BLOCK_M 时块 0 已写全部 T_NEW，因为更新循环覆盖 T_NEW 全量，不看 Q block）。**可立即改，需 GPU 验证正确性**。
3. **varlen bwd 的 GQA expand+sum 消除**（backward_varlen.cu:762-763,800-801）。当前为 `[T_K,H_Q,D]` 中间张量 + `at::sum_out`（2 次额外 kernel + 2×(T_K·H_Q·D·2) 字节流量 + 额外显存）。改为 dKV 相内用小 grid 原子累加（或让 dKV 相 grid 用 H_K 并在 kernel 内循环组，如 dense 的做法 backward.cu:350-358）。风险：中等（atomic 累加顺序不确定 → 与上游"deterministic=false"默认一致，但会改变数值轨迹）。**需要 GPU 验证 + 性能对比**。
4. **K/V tile 装载与 MMA 的流水线化（结构性）**：当前每个 block 内严格 `LOAD K → sync → SCORES → sync → SOFTMAX → sync → LOAD V → sync → PV`（forward.cu:153-204），无 double-buffer、无跨迭代重叠；Volta 没有 `cp.async`，只能靠寄存器暂存/双缓冲 smem 手工流水。收益潜力最大（掩盖 global latency），但寄存器/smem 压力大（smem 已近满）。**需要 GPU 验证，中高工作量**。
5. **python head-dim padding 粒度**：`_SUPPORTED_HEAD_DIMS=(16,32,64,128,256)`（flash_attn_v100/flash_attn_interface.py:23-32）→ 例如 head_dim=96/80 会被 pad 到 128（80→128 多算 60%）。上游同二进制支持所有 8 倍数的 D。机会：为常用 D（80/96/112/192/224）增加 `KernelConfig` 实例（每个 D 需要新的 BLOCK_M/BLOCK_N 表，forward.h:9-23、backward.h:9-37），或改进 padding 选择策略（pad 到 16 的倍数并新训小 D kernel）。风险：模板组合 × 编译时间暴涨；**需要 GPU 验证数值（padding 段 k=0 贡献 0，理论安全）**。
6. **smem 初始化减量**（forward.cu:125-128；backward 各相入口 backward.cu:140/317 等）。整块 74-92KB 清零 + row_max 二次填充，每次 kernel/每相执行；可选做法：只清必用子区（sO 行区、sP 全量等在 softmax 内本就会覆盖）。收益小-中（每次 ~11 条 st.shared.v4/线程），风险低。**可立即改**。
7. **split-KV（num_splits>1）实现**：decode 场景（T_Q 小、KV 长）现在只能用 1 个 CTA，grid 并行度 = B*H_Q，长 KV 完全串行；实现 split-KV（两段式 reduce lse）可显著加速长上下文 decode。工作量中-大，风险中。**需 GPU**。
8. **`is_causal` auto-off 以外的 decode 快速路径**：T_Q==1 时其实可把 causal 的 `window_right=0` + `block_max` 裁剪做得更激进（kvcache 已做 `window_right=0`，forward_kvcache.cu:464）。收益有限；**需 GPU**。
9. **dropout 路径 Philox 调用密度**：每个 4-wide 元素组一次 philox（softmax.h:106-138）；可考虑每线程一次性生成连续 4 个 counter 的流水（已是 v4 输出）。收益小；风险低；但 P0-1 修复合入时需要重构这段，顺带评估。

---

## 8. 无法静态确定的清单（could NOT determine statically）

1. **完整 kernel 的真实寄存器数与 spill**：torch（2.6.0）尚未装入 `/mnt/envs/v100`（`import torch` → ModuleNotFoundError），无法用真实 TU 编译 5 个 kernel 拿 `-Xptxas -v`；本报告 §5.2 只有合成热循环探测（[HYPOTHESIS] 级证据）。
2. **P1-2/P1-3/P1-4/P0-1 修复后的数值正确性**：全部需要 GPU 上跑 fwd/bwd 与 torch 参考对拍（尤其是 dropout 反向、paged 页布局、RoPE 位置）。
3. **MMA_884 的 mask 错位是否真的发生**（P2-4）：需要 GPU 或逐 lane 仿真验证；本报告只证明"代码常量与 fragment 文档布局不一致"。
4. **paged 跨页在真实部署中的出现率与危害**（P1-3）：取决于用户 page_block_size 与 block_table 物理布局，本机无法运行。
5. **上游 FA2 2.8.3 的精确对照**（dmask 语义、`leftpad_k` 在 varlen 的语义、RNG 流含 head/batch、head-dim padding 行为）：本地没有上游源码/wheel，只能用文档与代码推断（相关 HYPOTHESIS 已在正文标注）。
6. **`MMA_NATIVE` 构建（nvcuda::wmma）下 mask 常量与 wmma accumulator 布局是否一致**：默认构建用自研 wrapper（其 store/load 与 mask 常量出自同一作者），nvcuda::wmma 路径未验证（未编译该变体）。
7. **并发写同值 cache 行（P2-2）在真实硬件内存模型下的行为**：理论良性（同值 8B 自然对齐写），但无 GPU 无法实测确认（含 RoPE 读写重叠时的可见性）。
8. **性能结论（§7 收益排序）**：无 GPU，全部为结构推理，均需实测。
9. **`KERNEL_DEBUG` 构建路径**（include/debug.h 456 行的调试宏、setup.py:65-72 的 `-g --keep -Xptxas -v`）对正常构建是否完全惰性：只抽查了包含关系，未逐行读完 debug.h 全文件。

---

## 摘要（≤15 行）

1. 仓库是纯 sm_70 的 FA2 fork：单 fp16 精度、D∈{16,32,64,128,256}、512 线程、smem 74-92KB → **所有配置 1 CTA/SM（25% 占用）**，`__launch_bounds__(512,2)` 的 64-reg 上限纯属自伤（自然需求 111-126 regs，实测有 spill）。
2. 5 个 pybind 入口 + `flash_attn_2_cuda` 双模块名 + TORCH_LIBRARY（`flash_attn_v100`），调用链 python→wrapper→dispatch_attention_features(8 bool)→launcher<D>→kernel<D,flags>。
3. 功能面：causal/window/softcap/alibi/dropout/rope/GQA/varlen/paged 有实现；deterministic 被拒、split-KV(num_splits>1) 未实现、varlen 的 leftpad_k 被静默忽略、cache_seqlens 不更新。
4. **P0**：varlen+dropout 前/后向 RNG stride 不一致（fwd 用 max_seqlen_k、bwd 用 per-batch seqlen_k）→ 静默错梯度（forward_varlen.cu:233 vs backward_varlen.cu:259,490）。
5. **P1**：dense bwd 返回未初始化 dsoftmax_sum（backward.cu:691,719）；全 mask 行 P=1/输出=mean(V)（softmax.h:92-104，test.py:144 恰好跳过 M>N causal）；paged 读 tile 跨页（检查只有 %256）；kvcache 非 causal RoPE 位置全行共用（rotary.h:201-202）。
6. **P2** 亮点：kvcache 更新早退时序、每 block/head 重复写 cache、leftpad 忽略、MMA_884 mask 错位、dropout mask 与 head/batch 无关。
7. sm_70 合规：`ldmatrix/cp.async/bf16/wgmma/m16n8k8/...` 全零命中，且 `__CUDA_ARCH__ != 700` 显式 `#error`（两个 MMA 头）——源码级零超纲。
8. CUDA 12.6：源码无 12.9 依赖；">=12.9" 是作者环境策略（docs 装 12.9.1+torch2.10cu129），工作区 setup.py 已把闸门改为 12.6 默认（`FLASH_ATTN_MIN_CUDA` 可覆盖）并使无 GPU 构建可行；本机 nvcc 12.6.85 + conda gcc 12.4 实测可编译本仓库头文件（系统 gcc16 不可用）。
9. 验证强度：smem 常量经 nvcc `static_assert` 实测；寄存器经合成 TU 实测（HYPOTHESIS 级）；正确性结论全部为源码静态推导，均需 GPU 对拍确认（见 §8）。

# 08 — 内核红队复核：对 `02_repo_audit.md` 的对抗性验证（无 GPU，源码 + 探针）

> 对象：`/home/eculid/Documents/Github/flash-attention-v100`（工作区当前文件；行号与 02 审计一致；审计引用的 HEAD `55d685a` 与本地分支状态无关，本报告以工作区为准）。
> 方法：纯静态复核 + 与上游 FA 2.8.3（`/tmp/opencode/fa_upstream/flash_attn-2.8.3`）逐点对照 + 3 个探针（`/tmp/opencode/kernel_red/`：`lane_map.py` 逐 lane 布局代数；`rng_key.py` dropout RNG 键/组件代数；`sass_probe.cu` 用 nvcc 12.6 + cuobjdump 看 sm_70 SASS）。**无 GPU，未运行任何 CUDA 内核。**
> 结论摘要：P0-1 维持；P1-2/P1-3 维持；P1-1 成立但降级（Python autograd 层不可见）；**P1-4 证伪**（与上游语义逐字一致，是文档问题不是 bug）；P2-4 对 `-DMMA_884` 成立、默认构建经代数验证正确；**新增 3 项**：G-1（dense dropout 反向 philox 组件错位，覆盖面比 P0-1 更大）、G-2（dmask 8B 对齐 → 可能 crash）、G-3（bwd 忽略 seqused_k，仅裸绑定可见）。

## 0. 判定记号

- **CONFIRMED**：机制可由源码/代数量化证明，无需 GPU 即可定性。
- **REFUTED**：主张不成立（与上游对照/推导矛盾）。
- **NEEDS-GPU**：机制可疑但结论依赖运行时行为。
- 严重度 S0（静默错梯度/错结果且常见路径）/ S1（静默错结果但条件受限）/ S2（显式失败或罕见路径）/ S3（文档/ABI 卫生）。

## 1. 总表

| ID | 审计主张 | 判定 | 复核等级/置信 | 一句话结论 |
|----|----------|------|----------------|------------|
| P0-1 | varlen+dropout fwd/bwd RNG stride 不一致 | **CONFIRMED** | S0 / 高 | 维持 P0；另发现其组件映射还有 G-1，二者必须同修 |
| P1-1 | dense bwd 返回未初始化 dsoftmax_sum | **CONFIRMED**（降级） | S3 / 高 | 值确实是 `torch::empty` 垃圾，但 autograd 层丢弃 grads[3]，仅裸绑定可见 |
| P1-2 | fully-masked 行 P 全 1 → Out=mean(V) | **CONFIRMED** | S1 / 高 | 数学精确成立（NEG_INF 有限），仅"部分 tile 有效"的块能触发；修复点冷门但明确 |
| P1-3 | paged KV tile 跨页读错 | **CONFIRMED** | S1 / 高 | 上游的安全前提是 kBlockN∈{64,128,256} 整除 256；本仓 BLOCK_N=176/512 打破前提 |
| P1-4 | kvcache 非 causal 的 Q RoPE 位置 | **REFUTED** | S3 / 高 | 与上游文档/kernel/测试逐字一致；建议补 docstring，不改内核 |
| P2-4 | `-DMMA_884` mask lane 错位 | **CONFIRMED**（884 限定） | S2 / 高 | 默认 m16n16k16 经 256/256 双射验证正确；884 下 24/32 lane、128/256 槽位越出 8×8 子块 |
| G-1 | （新）bwd dropout philox 组件选择错位 | **CONFIRMED** | S0 / 高 | bwd 硬编码 x,y,z,w，仅当 (row·N)%4==0 才与 fwd 一致；dense N%4≠0 即错 |
| G-2 | （新）dmask 的 `st.global.v4.u16` 8B 对齐 | **CONFIRMED**（崩溃场景需 GPU 复现） | S2 / 高 | SASS 实测降为单条 `STG.E.64`；stride 非 4 倍数 → 错位 8B 全局写 |
| G-3 | （新）bwd 忽略 `seqused_k` | **CONFIRMED** | S3 / 高 | fwd 尊重 seqused_k（template.h:65-68），bwd 永远传 nullptr（:98,:327）；仅裸绑定可触发 |

## 2. 逐项裁定与战斗计划

### P0-1（CONFIRMED，维持）varlen+dropout 前/后向 RNG stride 不一致
- **证据**：fwd `include/softmax.h:109-117`（主循环）、`:154-158`（尾段）用 `addr_plx=(row_glob)*GLOBAL_N+(col)`；bwd `include/softmax.h:290-296` 同式。fwd varlen 传 `GLOBAL_N=max_seqlen_k`（`kernel/forward_varlen.cu:228-233`，含 `dmask` stride `H_Q*max_seqlen_k`）；bwd varlen 传 `block.seqlen_k`（`kernel/backward_varlen.cu:254-260`:259、`:485-491`:490）。dense 两侧都用 `N`（`forward.cu:177-182`；`backward.cu:223-229`、`:434-438`）。
- **量化**：`rng_key.py` 最小例（seqlen_k=[2,5]，max=5）batch0 的 row1 键即不同；任意 `seqlen_k[b]!=max_seqlen_k` 的 batch 上，"哪些列被丢弃"几乎逐元素错位 → dq/dk/dv 静默错误（loss 与梯度不一致）。bwd 注释"actual KV length for correct dropout RNG stride"与 fwd 的分工自相矛盾。
- **触发面**：所有 `varlen_fwd`+`varlen_bwd`+`p_dropout>0` 且长度参差的训练；`seqused_k` 令 fwd 截断时同样命中。注意 G-3：bwd 根本收不到 seqused_k。
- **修复（ABI 中立，2 行）**：把 bwd 两处 `block.seqlen_k` 改为 `max_seqlen_k`（`backward_varlen.cu:49` 已是内核入参；dKV 相 `:490` 同）。不要反向把 fwd 改成 `block.seqlen_k`：会改变已返回 dmask 的键空间（用户可见行为变化）。
- **最小 GPU 测试**：`B=2, seqs=[17,129], H=1, D=64, dropout=0.1`，固定 gen 种子；autograd 求 loss 后，用 fwd 返回的 `dmask` 手写参考：`P_drop=softmax(S)·mask/(1-p)`，与 flash 的 dq/dk/dv 对拍（先跑修复前，应当报大误差；修复后收敛）。当前 `tests/gpu_verify.py:252-279` 对 dropout 只查有限性、且不含 varlen 反向——该用例是纯增量覆盖。

### P1-1（CONFIRMED，降级 S3）dense backward 返回未初始化 `dsoftmax_sum`
- **证据**：`kernel/backward.cu:691` `torch::empty({B,H_Q,M})`；launcher 签名 `:494-514` 与 kernel 启动 `:563-569` 都没有 softmax_d 参数；`:719` 原样返回；只有 `M==0||N==0` 早退 `:699` 会 `zero_()`。对照 varlen 写入路径：`backward_varlen.cu:110`（早退清零）、`:145`（dLse 指针）、`:285-287`（dQ 相把 `sRowDot` 写出）。
- **降级理由（与审计的分歧）**：两个 Python autograd 都丢弃 `grads[3]`（`flash_attn_v100/flash_attn_interface.py:153-158` dense、`:332-337` varlen）→ 用户拿不到这个张量；只有直接调用 `flash_attn_v100_cuda.bwd(...)` 的裸绑定用户会看到垃圾/NaN。属"表面成功、语义空洞"，非静默错梯度。
- **修复**：dense dQ 相 `WMMA_GEMM_DOT_PRODUCT<...rowdot_dQ...>`（`backward.cu:159-162`）已经算出 `sRowDot`，在 dQ epilogue 后仿 varlen `:285-287` 写入即可：加 `float* softmax_d` 到 launcher/kernel，写 `softmax_d[block.lse_offset(H_Q,M)+tid]=sRowDot[tid]`。零新计算量。
- **最小 GPU 测试**：`bwd(...)[3]` 全有限且等于手算 `sum(O⊙dO)`（dense，M=N=64，dp=0）。

### P1-2（CONFIRMED，维持 S1）fully-masked 行 → P 全 1，Out=mean(V)
- **数学（精确）**：`include/softmax.h:92-104`：全 mask 行 `thread_max=NEG_INF`；`new_max=fmaxf(row_max, thread_max)=NEG_INF`；`exp_diff=exp(NEG_INF-NEG_INF)=exp(0)=1`；每元素 `exp(fmaxf(sS-new_max,-80))=exp(0)=1` → `sRowSum` 累加"被访问列数"，`Out=ΣV/count`。lse 写 `sRowMax+logf(fmaxf(sum,1e-24))=NEG_INF+log(count)≈-1e30`（`forward.cu:216-219`、`forward_varlen.cu:268-271`）。关键在于 `NEG_INF=-1e30f` 是**有限数**（`include/kernel.h:20`），故无 NaN、无 inf，全部是"看起来合理"的有限垃圾——审计的机制描述完全正确，我最初的 NaN 假设被推翻。
- **可达性（确认）**：需要"同一 Q-block 内后排行有有效 key、前行全被 mask"，使循环执行。dense causal M>N：`template.h:98-101` 只用块级 `max_key_pos` 裁剪，块内前行（`m < N-M`）全 mask 但块仍被执行。varlen causal `seqlen_k<seqlen_q` 同理；kvcache causal 且 `T_NEW>T_Q` 时 `seqlen_offset=T_Q-T_NEW<0` 也命中（其 lse 有 `:291` 的零和 guard，**但 Out 没有**）。整块无 key 时循环跳过：`sO`/`sRowSum` 初值分别为 0（`gemm_smem.h:59-67` INIT_SMEM 零填充）→ Out=0×rcp(1e-24)=0 正确，所以只有"部分命中"的块出错。
- **触发面**：causal M>N 的前若干行（dense/varlen/kvcache）；`test.py:144-145` 恰好 `continue` 掉该组合，仓库任何测试都测不到。
- **修复（fwd 一处）**：在 `include/softmax.h` 对 `new_max==NEG_INF` 的行把 P 全部置 0：主循环 `:99-102` 与尾段 `:148` 的 `e=__expf(fmaxf(...))` 改为 `all_masked ? 0.0f : __expf(...)`，其中 `const bool all_masked = (new_max == NEG_INF);`。则 `sRowSum=0` → EPILOGUE `norm=__frcp_rn(fmaxf(0,1e-24))=1e24`（`gemm_smem.h:196-198` 的 fmaxf 保护）、`Out=0×1e24=0`；lse 已是 `-1e30` 哨兵。部分 mask 的行 `new_max` 有限，`exp(NEG_INF-new_max)=0` 不受影响。**不要**改 `sRowSum`/epilogue 的 1e-24 保护（见 §4）。bwd 无需改：`softmax.h:282-285` 的 `sS==NEG_INF` 分支已把 P 置 0。
- **最小 GPU 测试**：dense `M=8,N=4,D=16,causal`（单 tile，D=16 的 BLOCK_M=16）→ 断言 `out[0:4]==0`、`lse[0:4]<=-1e29`、`out[4:8]` 正常；varlen `seqs_q=[8],seqs_k=[4]` 同断言；kvcache 同构。

### P1-3（CONFIRMED，维持 S1）paged KV 跨页读错
- **证据**：读路径每 tile 只取一次页号：`forward_varlen.cu:181-196` + 连续读 `:203-206`；`forward_kvcache.cu:199-215` + `:222-225`。门槛仅 `page_block_size % 256 == 0`（`:436`、`:486`）。写路径逐行查页（`rotary.h:62-71`）安全，与审计一致。
- **上游对照（本轮新增）**：上游只查 `%256`（`flash_api.cpp:582-583`），但上游 kBlockN 恒为 256/128/64（`flash_fwd_launch_template.h:168`），**任意 256 倍数的页都整除 kBlockN**，所以上游检查是充分的。本仓 BLOCK_N（`forward.h:9-22`）为 D=16→512、32→256、64→128、**128→176**、256→64：D=128 的 176 不整除任何 2 的幂页（256/512/1024/2048 全覆盖失败；理论最小安全页 2816）；D=16 在 page=256 与 768 失败，需 512 整除。物理页乱序时静默错 K/V；跨到 cache 张量末尾可能越界读。
- **修复（推荐分段加载，~15 行）**：paged 分支拆两段：`rows0=min(valid_kv_rows, block_page-page_offset)`，从 `phys0=block_table[bt+page_idx]` 读 rows0 行；若 `rows0<valid_kv_rows`，从 `phys1=block_table[bt+page_idx+1]` offset 0 读剩余行到 `sK/sV + rows0*D_STRIDE`（两次 `WMMA_GEMM_LOAD_TILE`，第二次 `VALID_ROWS=valid_kv_rows-rows0`）。`forward_varlen.cu` 与 `forward_kvcache.cu` 两处同改。保守熔断（D=128 会被全禁，仅作临时后退）：`TORCH_CHECK(page_block_size % BLOCK_N == 0)`。
- **最小 GPU 测试**：`page=256, D=16, B=1, T_Q=1, KV=600`，`block_table=[[3,1,5,0,...]]`（故意乱序+跨页 tile）；对拍参考 attention（非 paged 展平）；再加 `D=128, page=1024` 同构。当前无任何 paged 测试。

### P1-4（REFUTED）kvcache 非 causal 的 Q RoPE 位置
- 审计主张"非 causal 时全行 pos=cache_seqlen 是 bug"。**上游逐字实现同一语义**：上游 docstring `flash_attn_interface.py:1495-1500`（"If not causal and not local, the query @q will be rotated ... at indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens)"）；上游 kernel `flash_fwd_kernel.h:800-801`（非 causal 时 cos/sin 行 stride 置 0）；上游测试 `tests/test_flash_attn.py:2017-2032` 的参考正是把 q 重排成 `b 1 (s h) d` 后按 `cache_seqlens` 旋转（seq 维=1 → 全行同位置）。本仓 `rotary.h:177,201-202` 与之同构；`forward_kvcache.cu:463` 的 `T_Q==1 → causal=false` 无害（单行时 `start_q=0`，causal 与非 causal 的 `pos=cache+leftpad` 相同）。
- **动作（不改内核）**：本仓 `flash_attn_interface.py:411-413` 的 kvcache docstring 只有一行，应补上"非 causal 全局同位置"的语义说明 + 逃生通道（用户自行预旋转 q/k、不传 rotary_cos/sin；launcher 只在传 cos/sin 时才要求同时给 k，`:567-568`）。若未来要支持"真正的双向 prefill 逐 token RoPE"，那是新特性（与上游不兼容），需另立需求。

### P2-4/F（CONFIRMED，884 限定；默认构建正确）mask lane 映射
- **默认 m16n16k16：正确（本轮新证据）**。`lane_map.py` 对 `mat_mul.h:54-55,86-99,128-129` 的 `(row,col)` 公式与 `mma_m16n16k16.h:494-530` store 的 `r_base/c_base/x[i]` 分解逐 (lane,i) 比对：**256/256 槽位完全一致且构成 16×16 双射**（无复制竞争、每元素恰好被判定一次）。故 causal/window/alibi/softcap 全路径在默认构建下无 lane 错位。
- **`-DMMA_884`：错**。`mma_m8n8k4.h:293-321` 布局是 `r0=(l&1)+((l>>4)&1)*4, c0=l&2`（8×8 子块）；mat_mul 公式多出 `((l>>2)&1)*8`（行）与 `((l>>3)&1)*8`（列）。量化：**128/256 槽位行错、128/256 列错；24/32 lane 至少一维错；被错算的行/列恰有半数越出 8×8 子块（≥8）** → `in_bounds=false` 把块内有效元素打成 NEG_INF（不只是 mask 放错位置）。这也解释了为何 P2-4 的"[HYPOTHESIS]"可以直接升级为 CONFIRMED（代数确定性）。
- **`-DMMA_NATIVE`**：`nvcuda::wmma` 的 accumulator 片段布局是 implementation-defined，静态不可判 → NEEDS-GPU；在验证前视同与 mat_mul 常量不匹配。
- **修复**：把 lane→(row,col) 基址抽成后端提供的小函数/常量（如每个 mma_*.h 定义 `acc_row_base(lane)/acc_col_base(lane)`，mat_mul 调用），或至少 `#if defined(MMA_884) || defined(MMA_NATIVE) → #error`（当前 mask 层只支持默认后端）。`setup.py:118-122` 两者都是显式环境变量 opt-in，默认构建不受影响。
- **最小 GPU 测试**：`-DMMA_884` 构建跑 causal 对拍（D=64, M=64, N=64）即暴露；理想是补一个 CPU 端 lane-map 单测（直接跑 `lane_map.py` 的逻辑入 CI）。

### G-1（新发现，CONFIRMED，S0）bwd dropout philox 组件选择错位
- **机制**：fwd 每个 4 宽列的 philox 键是 `addr>>2`，元素 k 取 `(addr+k)&3` 选中的字（`softmax.h:114-117`），对任意 stride 自洽。bwd 用同一键（`:290-291`）**却硬编码** `pd0←rng.x, pd1←rng.y, pd2←rng.z, pd3←rng.w`（`:293-296`），等价于假设 `addr%4==0`。列部分 `col=4·m` 恒 4 对齐，所以是否错完全取决于 `(row_glob·GLOBAL_N)%4`。
- **量化（`rng_key.py`）**：dense 用 `N`：N%4=0 时全对；N=127 时 95/127 行取错组件、N=130 时 65/130、N=129 时 96/129。fwd 与 bwd 的"保留/丢弃"图案在这些行上不一致 → 静默错梯度。varlen 侧即使把 P0-1 的 stride 统一为 `max_seqlen_k`，只要 `(q_base+start_q)·max_seqlen_k %4 != 0` 依旧错。
- **触发面**：**dense（主 API！）`flash_attn_func(..., dropout_p>0)` 只要 seqlen_k 不是 4 的倍数**——比 P0-1 的 varlen 条件常见得多，是审计漏掉的最大单项。
- **修复（与 fwd 同构，改 softmax.h:293-296 一处）**：
  ```cpp
  const uint32_t c0 = static_cast<uint32_t>(addr_plx) & 3;
  const uint32_t r[4] = {rng.x, rng.y, rng.z, rng.w};
  if (in0) pd0 = (r[(c0 + 0) & 3] <= drop_thr) ? (p0 * rp_dropout) : 0.0f;
  if (in1) pd1 = (r[(c0 + 1) & 3] <= drop_thr) ? (p1 * rp_dropout) : 0.0f;
  if (in2) pd2 = (r[(c0 + 2) & 3] <= drop_thr) ? (p2 * rp_dropout) : 0.0f;
  if (in3) pd3 = (r[(c0 + 3) & 3] <= drop_thr) ? (p3 * rp_dropout) : 0.0f;
  ```
- **最小 GPU 测试**：dense `M=128,N=127,D=64,dropout=0.1`，与 P0-1 相同思路（用 fwd 的 dmask 构造参考 P_drop 对拍 dq/dk/dv）；该尺寸同时覆盖 P0-1（若换成 varlen 版本）。
- **顺序建议**：P0-1 与 G-1 必须**同一个 PR**修（同一代码路径的两个缺陷），修完跑同一个 dropout 数值对拍 > 4 组尺寸（N%4=0/1/2/3）× (dense, varlen)。

### G-2（新发现，CONFIRMED（错位）/ NEEDS-GPU（是否 fault），S2）dmask 8B 错位全局写
- **机制**：`softmax.h:129-137` 对 dmask 用 `st.global.v4.u16`（8B）。`sass_probe.cu` 在 sm_70 上实测该指令被 ptxas 降为**单条 `STG.E.64`**——要求地址 8B 对齐，即元素下标 `idx=row·STRIDE_GMEM_MASK+start_kv+4·addr` 必须 ≡0 (mod 4)。`start_kv`/`4·addr` 恒 4 对齐，所以条件化为 `STRIDE_GMEM_MASK%4`：dense `=N`（`forward.cu:182`）；varlen `=H_Q*max_seqlen_k`（`forward_varlen.cu:233`）。N=127 时 `idx mod 4` 随行在 {0,3,2,1} 轮转 → 3/4 的行错位 8B 写。尾段已是标量写（`:164-167`）不受影响。
- **触发面**：`dropout_p>0 且 return_attn_probs=True`（dmask 仅此时分配：`forward.cu:397-401`、`forward_varlen.cu:527-532`）+ stride 非 4 倍数。典型调用 `flash_attn_func(q,k,v,dropout_p=0.1,return_attn_probs=True)` 且 `seqlen%4!=0` → 运行时报 misaligned address（PTX 对非自然对齐向量访问为 UB，硬件上即 fault）。属于"显式失败"而非静默错值；若用户从不请求 dmask 则无感。
- **修复（二选一，均局部）**：① `softmax.h` 写入处加运行时分支：`(STRIDE_GMEM_MASK&3)==0` 走现有 v4，否则 4×`st.global.u16`（指令开销仅 dmask 路径）；② 分配侧把行 stride 垫到 4 的倍数（`N_pad=(N+3)&~3`），把 `N_pad` 作为 STRIDE 传入、返回 `dmask[..., :N]` 视图。
- **最小 GPU 测试**：dense `M=64,N=127,D=32,dropout=0.5,return_attn_probs=True`（当前应直接抛 misaligned address / 断言）；修复后断言 dmask 形状 `[B,H,M,N]` 且取值 ±1 与 P_drop 符号一致。

### G-3（新发现，CONFIRMED，S3）varlen bwd 忽略 seqused_k
- **证据**：fwd 尊重 `seqused_k`（`template.h:65-68` 截断 `seqlen_k`；`forward_varlen.cu:36,91,337,356,458-464`）；bwd 两相一律 `nullptr // not used in backward`（`backward_varlen.cu:98,327`），launcher 也无该参数 → 同一输入下 fwd 只看前 `seqused_k[b]` 个 key，bwd 按全长计算 → dq/dk/dv 静默错（dK 还含 fwd 从未看见的 key 的贡献）。
- **触发面**：只有裸 `flash_attn_v100_cuda.varlen_fwd/bwd` 用户；Python autograd 恒 `seqused_k=None`（`flash_attn_interface.py:271-272`）不可触发。与 P0-1 修复无冲突（两内核都用 host 的 `max_seqlen_k` 做 RNG stride）。
- **修复**：给 varlen bwd 增加可选 `seqused_k`（ABI 扩展）并同样传入 BlockInfo；或在裸绑定层 `TORCH_CHECK(!seqused_k.has_value() || !requires_grad)` 显式拒绝。

## 3. 严重度 × 置信度 排序（修复优先级）

1. **G-1** — S0/高，dense+dropout 日常路径静默错梯度（N%4≠0 占多数长度）→ 与 **P0-1** 一起修。
2. **P0-1** — S0/高，varlen+dropout 静默错梯度 → 同上（2 行）。
3. **P1-3** — S1/高，paged+D=128（或 D=16 小页）静默错结果/潜在越界 → 分段加载。
4. **P1-2** — S1/高，causal M>N 顶部行的 Out 是 mean(V) → 一处 fwd 修复。
5. **G-2** — S2/高（错位确证），dropout+return_attn_probs 且 stride 非 4 倍 → 会崩溃，值得尽早修但非静默。
6. **P2-4/F** — S2/高，仅 `-DMMA_884` 显式构建；默认构建已证正确 → 加编译期 disable 或后端化 lane 基址。
7. **P1-1** — S3/高，垃圾返回值只在裸绑定；顺手补 `softmax_d` 写回（零新计算）。
8. **G-3** — S3/高，裸绑定 seqused_k 场景；补 ABI 或拒绝。
9. **P1-4** — REFUTED，仅补文档（kvcache RoPE 非 causal 语义 + 预旋转逃生通道）。

## 4. "无 GPU 不要动"清单（每项附理由/必须配套的验证）

1. `include/mat_mul.h:54-55,86-99,128-129` mask/lane 常量：默认后端**已证正确**，为"修 884"而乱改会波及全部 5 个 D × causal/window/alibi/softcap。要改先跑 `lane_map.py` + GPU causal 对拍。
2. `include/template.h:98-111,180-193` BlockInfo 裁剪：P1-2/P1-3 修复不得改变它；任何改动都要重测 causal M>N/M<N/M=N 与 window 左右边界。
3. `include/softmax.h:85-190, 250-339`：fwd/bwd 数值核心，P0-1/G-1/P1-2/G-2 四个修复都落在这里——合入前必须有一次完整数值对拍（dense/varlen/kvcache × dropout 0/0.1 × causal 0/1 × N%4=0..3）。
4. `include/gemm_smem.h:196-198` 的 `fmaxf(SMEM_DOT[row],1e-24f)` 与 `INIT_SMEM` 零填充：P1-2 的"只修 P=0、保留 norm guard"路线依赖二者（`0×1e24=0`）；norm guard 不要改成无保护的 `1/sum`，也不要在 P=0 修复落地前改动它（否则 fully-masked 行的 Out 会从 mean(V) 变成 inf/NaN）。
5. `include/rotary.h`（tile loader + swizzle + KV 更新）：P1-4 只改文档不动它；任何 tile 布局改动需全部 D 的 fwd 对拍。
6. `kernel/forward_kvcache.cu` 早退/更新顺序（审计 P2-1）：把 `UPDATE_KVCACHE` 前移会改变写语义，需 kvcache append 测试（含 `seqlen_k==0`、leftpad、paged）。
7. P0-1 的"反向修法"（fwd 改 `block.seqlen_k`）：会改变所有已跑 dropout 的键空间与返回的 dmask（用户可见行为），需全量再基线化；默认走 bwd→`max_seqlen_k`，不要顺手改 fwd。
8. `-DMMA_884` / `-DMMA_NATIVE` 生产构建：在 lane 基址后端化并通过 GPU mask 对拍前，保持关闭（`setup.py:118-122`）。

## 5. 覆盖缺口与建议的最小 GPU 回归集

- 既有覆盖：`test.py` dense fwd/bwd（dropout=0，`:144-145` 跳过 causal M>N）；`tests/gpu_verify.py` varlen **仅 fwd**（`:327`），dense bwd dropout 仅查有限性（`:273-279`），kvcache 无 rotary/paged/反向（`:219-242`）。
- 建议新增（按本报告最小集）：① dense `M=128,N=127,dp=0.1` 的 dmask-consistent 反向对拍（G-1）；② varlen `seqs=[17,129]` 同款（P0-1+G-1）；③ causal `M=8,N=4` 的 Out=0/lse 哨兵（P1-2）；④ paged `page=256`、`D∈{16,128}`、乱序 block_table 对拍（P1-3）；⑤ `dp=0.5, return_attn_probs=True, N=127` 崩溃复现（G-2）；⑥ 裸绑定 `bwd(...)[3]` 与手算 row_dot 对比（P1-1）；⑦ 裸绑定 varlen `seqused_k` 与 bwd 的差异复现（G-3）。

## 6. 探针与复现命令（全部在 `/tmp/opencode/kernel_red/`，未写入仓库）

```
python3 lane_map.py    # 默认 m16n16k16: 256/256 双射一致; MMA_884: 128+128/256 槽位错、24/32 lane 错、半数越出 8x8
python3 rng_key.py     # varlen [2,5]/max5 键不一致; dense N=127: 95/127 行组件错位
nvcc -arch=sm_70 -O3 -cubin sass_probe.cu -ccbin=/mnt/envs/v100/bin/x86_64-conda-linux-gnu-g++
# 本机没有独立的 cuobjdump，用 triton wheel 缓存里的：
/mnt/.cache/uv/archive-v0/Uut6HUvniXbXIf8P/triton/backends/nvidia/bin/cuobjdump -sass sass_probe.cubin
#   st.global.v4.u16 -> STG.E.64 ; ld.global.v4.u32 -> LDG.E.128
```

---

**给执行者的最短行动序列**：`G-1+P0-1（同 PR，softmax.h + backward_varlen.cu 两处） → 数值对拍① ②；G-2 → ⑤；P1-2 → ③；P1-3 → ④；P2-4 加保护/后端化；P1-1、G-3 顺手补；P1-4 只补 docstring。`

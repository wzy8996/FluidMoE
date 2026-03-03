# FluidMoE 理论分析（中文推导版）

本文档面向当前代码版本，给出 `Baseline` 与 `FluidMoE` 的理论计算量、通信量与时间模型推导，并明确说明哪些结论是“代码确定的”，哪些是“建模假设”。

重点目标：

1. 给出 `Baseline` 单层训练（forward + backward）的计算量与通信量推导过程。
2. 给出 `DP AllReduce (AR)` 的通信量建模方法，并说明你当前代码里的 AR 机制。
3. 说明 `FluidMoE` 在**不考虑重叠调度**时，其主项计算量与通信量与 `Baseline` 一致。
4. 给出后续进行“理论加速比分析”时可用的理想化时间模型（先忽略 `n_msg * tau` 与调度开销）。

## 1. 分析范围与代码口径

本文档对应当前实现的关键变化（已体现在代码中）：

1. Attention 使用 PyTorch 原生 `SDPA`，支持原生 `GQA`（通过 `enable_gqa=True`）。
2. Attention backward 不再走 `attention_score_backward()` 的 Python 层手动重算，而是：
   - 前向保存 SDPA 局部计算图；
   - 反向直接对保存的 `attn_out_bf` 调用 `torch.autograd.grad(...)`。
3. `tests/baseline.py` 已补齐 `router backward`（不再是 `grad_router_weight = 0` 的简化版）。
4. `tests/baseline.py` 已保存 `attn_out_sp`，避免 backward 中额外一次 `hp2sp`。

因此本文档中的 `Baseline` 与 `FluidMoE` 理论公式使用的是**当前代码口径**，而不是旧版本简化口径。

## 2. 模型结构与执行流程（单层）

单层结构（与 `TransformerLayerFunction` 一致）：

1. `LN1`
2. Attention block
   - `QKV projection`
   - CP 通信（`sp2hp`）
   - `SDPA`
   - CP 通信（`hp2sp`）
   - `Output projection`
   - Residual
3. `LN2`
4. MoE block
   - `Router`
   - EP 通信（`Dispatch`）
   - Expert `FC1 + Act + FC2`
   - EP 通信（`Combine`）
   - 概率加权 + 恢复顺序 + sum(top-k)
   - Residual

### 2.1 Baseline 单层训练流程（概念上）

Forward：

1. `LN1`
2. `QKV Linear`
3. CP `sp2hp`
4. `SDPA`
5. CP `hp2sp`
6. `OutProj`
7. `LN2`
8. `Router`
9. EP `Dispatch`
10. Experts `FC1 + Act + FC2`
11. EP `Combine`

Backward：

1. MoE backward（Combine backward -> Expert backward -> Dispatch backward）
2. Router backward（完整链路）
3. `LN2 backward`
4. Attention backward
   - `OutProj dX/dW`
   - CP `sp2hp`（grad attn out）
   - `SDPA backward`（保存图 + `autograd.grad`）
   - CP `hp2sp`（grad qkv）
   - `QKV dX/dW`
5. `LN1 backward`
6. （若 `DP>1`）DP AR（参数梯度归约）

### 2.2 FluidMoE 单层训练流程（概念上）

与 Baseline 的算子集合相同，但执行顺序被 scheduler 重排并分成 4 个 backward region（并可重叠）：

1. `moe_combine`
2. `moe_dispatch`
3. `attn_proj`
4. `attn_qkv`

另外：

- `dW` 计算会被延迟成任务，在 AllToAll 窗口中执行；
- `DP AR` 可分块、可在 AllToAll 空隙中由独立线程执行。

重要结论（后续会证明）：

- **不考虑重叠与调度开销时**，`FluidMoE` 的主项计算量与通信量与 Baseline 相同。

## 3. 符号定义（兼容 MHA / GQA）

- `s`: 总序列长度（不乘 batch）
- `b`: micro-batch size
- `h`: hidden size
- `h_ff`: MoE FFN hidden size
- `n_h`: Q 头数
- `n_kv`: KV 头数（GQA 时 `< n_h`）
- `d = h / n_h`: head dim
- `E`: 总专家数
- `K`: top-k
- `P`: 并行度（这里取 `CP = EP = P`）
- `D`: DP 并行度（用于 AR 建模）
- `beta`: 激活/普通 bf16 参数的每元素字节数（通常 `beta = 2`）
- `beta_r`: router 参数梯度每元素字节数（当前实现中 `router_weight` 为 fp32，通常 `beta_r = 4`）

### 3.1 GQA 关键中间量

定义：

$$
q\_{per\_group} = \frac{n_h}{n_{kv}}
$$

$$
h_{qkv} = h\left(1 + 2\frac{n_{kv}}{n_h}\right)
$$

解释：

- Q 总宽度始终为 `h`
- K/V 总宽度在 GQA 下缩小
- 因此 QKV 总投影宽度从 MHA 的 `3h` 变为 `h_qkv`

特例：

- MHA：`n_kv = n_h`，所以 `h_qkv = 3h`

## 4. 计算量计数规则与通信计数口径

### 4.1 FLOPs 计数规则

矩阵乘：

$$
[M,K] \times [K,N] \Rightarrow 2MKN
$$

本文主项只统计：

- GEMM 主项
- Attention 的 `s^2` 主项（`QK^T` 与 `Attn@V`，以及其 backward 对应项）

忽略项（通常较小）：

- LayerNorm
- 激活函数
- softmax / top-k
- scatter / index / reshape / permute 的纯索引开销

### 4.2 通信计数口径

本文通信量默认统计为：

- **每个 rank 的发送量（send bytes）**

对于均匀 AllToAll/P2P 交换，本地数据中约有 \((P-1)/P\) 发送到其他 rank，因此发送量可写为：

$$
\text{send bytes} = \text{local elements} \cdot \frac{P-1}{P} \cdot \text{element bytes}
$$

换成 `per-rank` 写法时常出现 \((P-1)/P^2\) 系数（因为 local elements 自身含 `1/P`）。

## 5. Baseline：Attention Forward 推导（per-rank）

### 5.1 QKV Projection 计算量

每个 rank 的本地 token 数：

$$
\frac{sb}{P}
$$

QKV 投影矩阵乘：

$$
\left[\frac{sb}{P}, h\right]\times[h, h_{qkv}]
$$

因此 FLOPs：

$$
F^{attn,fwd}_{qkv} = 2 \cdot \frac{sb}{P}\cdot h \cdot h_{qkv}
$$

### 5.2 `sp2hp` 通信量（QKV）

本地 QKV 元素数：

$$
\frac{sb}{P}\cdot h_{qkv}
$$

发送量：

$$
B^{attn,fwd}_{sp2hp}
=
\frac{P-1}{P}\cdot \frac{sb}{P}\cdot h_{qkv}\cdot \beta
=
\frac{P-1}{P^2}sbh_{qkv}\beta
$$

### 5.3 SDPA 主项计算量（QK^T + AV）

尽管 GQA 下 `K/V` 头数较少，但每个 Q 头仍需要生成 `s x s` 的注意力分数并完成加权求和，因此 `s^2` 主项仍按 Q 头数 `n_h` 计。

`QK^T`：

$$
F_{QK} = 2 \cdot b \cdot \frac{n_h}{P}\cdot s \cdot s \cdot d
=
\frac{2bhs^2}{P}
$$

`Attn @ V`：

$$
F_{AV} = \frac{2bhs^2}{P}
$$

所以 SDPA forward 主项：

$$
F^{attn,fwd}_{sdpa} = \frac{4bhs^2}{P}
$$

### 5.4 `hp2sp` 通信量（attn output）

本地 attn 输出元素数（HP 侧）：

$$
sb \cdot \frac{h}{P}
$$

发送量：

$$
B^{attn,fwd}_{hp2sp}
=
\frac{P-1}{P^2}sbh\beta
$$

### 5.5 Output Projection 计算量

$$
\left[\frac{sb}{P}, h\right]\times[h,h]
\Rightarrow
F^{attn,fwd}_{proj}
=
2\cdot \frac{sb}{P}\cdot h^2
$$

### 5.6 Attention Forward 汇总

$$
F^{attn,fwd}
=
\frac{2sbh(h_{qkv}+h)+4bhs^2}{P}
$$

$$
B^{attn,fwd}
=
\frac{P-1}{P^2}sb(h_{qkv}+h)\beta
$$

MHA（`h_qkv = 3h`）时：

$$
F^{attn,fwd}_{MHA}
=
\frac{8sbh^2+4bhs^2}{P}
$$

$$
B^{attn,fwd}_{MHA}
=
\frac{P-1}{P^2}4sbh\beta
$$

## 6. Baseline：Attention Backward 推导（per-rank，当前代码口径）

当前版本的 `baseline` 与 `FluidMoE` 主实现都已采用：

- 前向保存 SDPA 局部计算图
- backward 直接 `torch.autograd.grad(attn_out_bf, (q,k,v), grad_output)`

因此这里**不再包含 Python 层手动重算一次 SDPA forward** 的计算量。

### 6.1 OutProj backward（dX + dW）

- dX 一个 GEMM
- dW 一个 GEMM

$$
F^{attn,bwd}_{proj}
=
\frac{4sbh^2}{P}
$$

### 6.2 `sp2hp` 通信量（grad attn out）

宽度为 `h`：

$$
B^{attn,bwd}_{sp2hp}
=
\frac{P-1}{P^2}sbh\beta
$$

### 6.3 SDPA backward 主项（理论主项近似）

当前代码级显式路径不再包含 Python 层重算，但 `SDPA` backward 的主项仍包含多个 `s^2 d` 级矩阵乘。用主项近似：

$$
F^{attn,bwd}_{sdpa}
\approx
\frac{8bhs^2}{P}
$$

说明：

- 这是主项近似；
- 底层 fused kernel（如 FlashAttention）内部可能有 tile 级重算，但属于 kernel 实现开销，不影响此处主项口径。

### 6.4 `hp2sp` 通信量（grad QKV）

QKV backward 通道宽度为 `h_qkv`：

$$
B^{attn,bwd}_{hp2sp}
=
\frac{P-1}{P^2}sbh_{qkv}\beta
$$

### 6.5 QKV backward（dX + dW）

- dX 一个 GEMM
- dW 一个 GEMM

$$
F^{attn,bwd}_{qkv}
=
\frac{4sbhh_{qkv}}{P}
$$

### 6.6 Attention Backward 汇总（当前代码）

$$
F^{attn,bwd}
\approx
\frac{4sbh(h_{qkv}+h)+8bhs^2}{P}
$$

$$
B^{attn,bwd}
=
\frac{P-1}{P^2}sb(h_{qkv}+h)\beta
$$

## 7. Baseline：MoE Forward / Backward 推导（per-rank）

### 7.1 MoE Forward

#### (1) Router

$$
\left[\frac{sb}{P}, h\right]\times[h, E]
\Rightarrow
F^{moe,fwd}_{router}
=
\frac{2sbhE}{P}
$$

#### (2) Dispatch 通信量

每 token top-k 扩展后，本地 token 数约为 `sbK/P`，每 token 宽度 `h`：

$$
B^{moe,fwd}_{dispatch}
=
\frac{P-1}{P^2}sbKh\beta
$$

#### (3) Experts：FC1 + FC2

在均匀路由近似下，每个 rank 处理的 expert token 总数为 `sbK/P`。

FC1：

$$
2\cdot \frac{sbK}{P}\cdot h \cdot h_{ff}
$$

FC2：

$$
2\cdot \frac{sbK}{P}\cdot h_{ff} \cdot h
$$

合计：

$$
F^{moe,fwd}_{expert}
=
\frac{4sbKhh_{ff}}{P}
$$

#### (4) Combine 通信量

$$
B^{moe,fwd}_{combine}
=
\frac{P-1}{P^2}sbKh\beta
$$

#### (5) MoE Forward 汇总

$$
F^{moe,fwd}
=
\frac{2sbhE + 4sbKhh_{ff}}{P}
$$

$$
B^{moe,fwd}
=
\frac{P-1}{P^2}2sbKh\beta
$$

### 7.2 MoE Backward（当前 Baseline 与 FluidMoE 完整口径）

#### (1) Expert backward 主项

主项近似包含：

1. FC1 重算
2. FC2 dX
3. FC2 dW
4. FC1 dX
5. FC1 dW

每项约为：

$$
\frac{2sbKhh_{ff}}{P}
$$

合计：

$$
F^{moe,bwd}_{expert}
=
\frac{10sbKhh_{ff}}{P}
$$

#### (2) Router backward + router dW

完整 router backward 包含概率链路反传与线性层 dX / dW。主项近似由两次 router 线性相关 GEMM 给出：

$$
F^{moe,bwd}_{router}
\approx
\frac{4sbhE}{P}
$$

#### (3) MoE backward 通信量

与 forward 对称（Combine backward + Dispatch backward）：

$$
B^{moe,bwd}
=
\frac{P-1}{P^2}2sbKh\beta
$$

#### (4) MoE Backward 汇总

$$
F^{moe,bwd}
\approx
\frac{10sbKhh_{ff}+4sbhE}{P}
$$

$$
B^{moe,bwd}
=
\frac{P-1}{P^2}2sbKh\beta
$$

## 8. Baseline：单层训练总计算量与通信量（per-rank）

### 8.1 Attention 总量

$$
F^{attn,train}
=
F^{attn,fwd}+F^{attn,bwd}
\approx
\frac{6sbh(h_{qkv}+h)+12bhs^2}{P}
$$

$$
B^{attn,train}
=
B^{attn,fwd}+B^{attn,bwd}
=
\frac{P-1}{P^2}2sb(h_{qkv}+h)\beta
$$

### 8.2 MoE 总量

$$
F^{moe,train}
=
F^{moe,fwd}+F^{moe,bwd}
\approx
\frac{6sbhE + 14sbKhh_{ff}}{P}
$$

$$
B^{moe,train}
=
B^{moe,fwd}+B^{moe,bwd}
=
\frac{P-1}{P^2}4sbKh\beta
$$

### 8.3 单层总量

$$
F^{layer,train}_{baseline}
\approx
\frac{6sbh(h_{qkv}+h)+12bhs^2 + 6sbhE + 14sbKhh_{ff}}{P}
$$

$$
B^{layer,train}_{baseline}
=
\frac{P-1}{P^2}sb\left[2(h_{qkv}+h)+4Kh\right]\beta
$$

MHA（`h_qkv = 3h`）时：

$$
F^{layer,train}_{baseline,MHA}
\approx
\frac{24sbh^2+12bhs^2 + 6sbhE + 14sbKhh_{ff}}{P}
$$

$$
B^{layer,train}_{baseline,MHA}
=
\frac{P-1}{P^2}4sbh(2+K)\beta
$$

## 9. DP AllReduce（AR）建模：代码机制与通信量

你当前代码中的 `AR` 方式不是简单的“主线程同步 all_reduce”，而是：

1. `ar_thread + ar_stream` 独立执行；
2. 使用独立 `ar_group`（单独 NCCL communicator）；
3. backward 期间按参数梯度分块提交（`_submit_ar_chunked`）；
4. `ar_thread` 等待 AllToAll 空档（`_a2a_idle`）后执行；
5. `finish_batch()` 期间对剩余 dW 会在末尾提交未分块 AR（当前 batch 的 AR 不在 `finish_batch()` 末尾等待）。

### 9.1 参与 AR 的参数集合（当前代码）

参与 DP AR（非 expert 参数）：

- `qkv_weight`
- `proj_weight`
- `router_weight`
- `ln1_weight`, `ln1_bias`, `ln2_weight`, `ln2_bias`

不参与 DP AR（expert 参数，EP 参数）：

- `moe_w1`, `moe_w2`

### 9.2 每层 AR 的参数梯度字节数（代码可确定）

定义每层需要做 AR 的参数梯度总字节数（每个 rank 本地持有）为 `G_{AR,layer}`。

这里先强调一个容易混淆的点：

- `G_{AR,layer}`：参数梯度本身的总字节数（代码可直接由参数形状和 dtype 确定）
- `B^{AR}`：网络上传输字节数（还要乘上 all-reduce 算法相关系数，如 ring 的 `2(D-1)/D`）

你当前代码里每层参与 AR 的参数、形状与字节数对应关系如下（按当前实现口径）：

1. `qkv_weight`
   - 形状：`[h_qkv, h]`
   - dtype：通常 bf16（梯度按参数 dtype 归并）
   - 字节数：

$$
G_{qkv} = h_{qkv} \cdot h \cdot \beta
$$

2. `proj_weight`
   - 形状：`[h, h]`
   - dtype：通常 bf16
   - 字节数：

$$
G_{proj} = h^2 \cdot \beta
$$

3. `router_weight`
   - 形状：`[h, E]`
   - dtype：当前实现中为 fp32
   - 字节数：

$$
G_{router} = hE \cdot \beta_r
$$

4. `ln1_weight`, `ln1_bias`, `ln2_weight`, `ln2_bias`
   - 4 个长度为 `h` 的向量
   - dtype：通常 bf16
   - 合计字节数：

$$
G_{ln} = 4h \cdot \beta
$$

因此，逐项求和得到每层 AR 参数梯度总字节数：

$$
G_{AR,layer} = G_{qkv} + G_{proj} + G_{router} + G_{ln}
$$

展开后即：

$$
G_{AR,layer}
=
h h_{qkv}\beta + h^2\beta + 4h\beta + hE\beta_r
$$

其中：

- 前三项（`qkv/proj/ln`）通常是 bf16 梯度（`beta = 2`）
- `router_weight` 为 fp32（`beta_r = 4`）

若直接写成常见取值形式：

$$
G_{AR,layer}
=
2(hh_{qkv}+h^2+4h)+4hE
$$

### 9.3 AR 的网络通信量（建模假设）

代码里调用的是 `dist.all_reduce(...)`，未在 Python 层显式指定 NCCL 算法，因此：

- **代码无法静态证明一定是 ring**
- `Ring / Tree / CollNet` 等由 NCCL 运行时决定

为了进行理论分析，可先采用 ring all-reduce 的常见近似（建模假设）：

$$
B^{AR}_{layer,ring}
=
2\frac{D-1}{D} \cdot G_{AR,layer}
$$

这里：

- `G_{AR,layer}` 是参数梯度字节数（代码确定）
- 系数 \(2\frac{D-1}{D}\) 是 ring all-reduce 的网络传输倍数（建模假设）

也就是说：

- 你可以从代码精确得到 `G_{AR,layer}`
- 但 `B^{AR}` 需要结合 all-reduce 算法假设（例如 ring / tree）来估算

### 9.4 代入当前 `tests/benchmark.py` 配置的 AR 数值（示例）

当前 `tests/benchmark.py` 默认配置：

- `h=4096`
- `n_h = n_kv = 32`（MHA，故 `h_qkv = 3h = 12288`）
- `E=8`
- `num_layers = 2`
- `dp_group = WORLD`，当 `world_size=2` 时可视为 `D=2`

每层参数梯度字节数：

1. `qkv_weight`: `12288 x 4096 x 2 = 100,663,296 B`
2. `proj_weight`: `4096 x 4096 x 2 = 33,554,432 B`
3. `ln1/ln2` weight+bias 共 4 个向量：`4 x 4096 x 2 = 32,768 B`
4. `router_weight`: `4096 x 8 x 4 = 131,072 B`

合计：

$$
G_{AR,layer} = 134,381,568\ \text{B}
$$

约等于：

- `134.38 MB`（十进制）
- `128.16 MiB`（二进制）

两层：

$$
G_{AR,total} = 268,763,136\ \text{B}
$$

当 `D=2` 且采用 ring 近似时，系数 \(2(D-1)/D = 1\)，因此每个 rank 的网络传输字节近似与 `G_{AR,total}` 相同：

$$
B^{AR}_{total,ring} \approx 268,763,136\ \text{B}
$$

注意：

- `ar_chunk_size`（当前 benchmark 设为 `16MB`）只改变 chunk 数与时序，不改变总字节数。

## 10. FluidMoE（不考虑重叠）与 Baseline 的等价性确认

### 10.1 计算量等价（主项）

`FluidMoE` 的 scheduler 会改变：

- 执行顺序
- 分块方式
- 计算/通信重叠关系
- dW 执行时机

但不会改变：

- QKV / Proj / Expert / Router 的 GEMM 主项
- SDPA 的 `s^2` 主项

因此在**不考虑重叠与调度开销**时：

$$
F^{layer,train}_{FluidMoE,\ no\ overlap}
=
F^{layer,train}_{baseline}
$$

### 10.2 CP/EP 通信量等价（主 payload）

`FluidMoE` 虽然使用：

- forward P2P round-robin
- backward chunked AllToAll

但总交换的激活元素数与 Baseline 相同，因此 CP/EP 主 payload 相同：

$$
B^{CP/EP}_{FluidMoE,\ no\ overlap}
=
B^{CP/EP}_{baseline}
$$

### 10.3 AR 通信量等价（参数字节层面）

`FluidMoE` 与 Baseline 使用同一组需要 DP AR 的参数（非 expert 参数），所以参数梯度总字节相同：

$$
G^{AR}_{FluidMoE} = G^{AR}_{baseline}
$$

若使用相同的 DP 组与相同的 all-reduce 算法假设（如 ring），则网络通信量建模也相同：

$$
B^{AR}_{FluidMoE} = B^{AR}_{baseline}
$$

区别只在于：

- `FluidMoE` 的 AR 可以被部分隐藏（和 A2A 空隙重叠）
- Baseline 通常更接近串行暴露

## 11. 理想化时间模型（用于相对理论加速分析）

为了做“相对理论加速效果”分析，可先采用主项模型，暂时忽略：

- `n_msg * \tau`（消息启动延迟）
- scheduler 线程/事件/同步开销
- Python / autograd bookkeeping 开销

### 11.1 Baseline（无重叠）

把一层总时间拆成三部分：

1. 计算时间（Attention + MoE）
2. CP/EP 通信时间
3. AR 通信时间

可写为：

$$
T^{layer}_{baseline}
\approx
\frac{F^{layer,train}_{baseline}}{\Pi}
+
\frac{B^{CP/EP}_{baseline}}{BW_{cp/ep}}
+
\frac{B^{AR}_{baseline}}{BW_{dp}}
$$

更精细时，可把 `BW_{cp}` 和 `BW_{ep}` 分开。

### 11.2 FluidMoE（理想重叠，忽略调度开销）

在理想重叠分析中，按 region 写成：

$$
T^{layer}_{fluid}
\approx
\sum_{r \in \{R1,R2,R3,R4\}}
\max\left(T^{comp}_r,\ T^{comm}_r\right)
+
T^{tail}
+
T^{AR}_{visible}
$$

其中：

- `T^{AR}_{visible}` 是 AR 未被隐藏的剩余可见时间。

如果进一步用“可隐藏 AR 时间预算” `T^{AR}_{hide}` 建模：

$$
T^{AR}_{visible}
=
\max\left(0,\ \frac{B^{AR}}{BW_{dp}} - T^{AR}_{hide}\right)
$$

### 11.3 理论加速比（主项模型）

$$
Speedup_{theory}
=
\frac{T_{baseline}}{T_{fluid}}
$$

该速度提升是：

- 忽略消息延迟与调度开销的理想上界趋势；
- 用于解释“为什么 FluidMoE 会快”以及“瓶颈在哪些区域”。

## 12. 使用建议（做实验对照时）

1. 先用本文公式得到 `Baseline` 与 `FluidMoE(no overlap)` 的相同主项，作为正确性检查。
2. 再引入 region 重叠模型，计算理想加速比上界。
3. 最后用实测与理论差距解释以下因素：
   - `n_msg * \tau`
   - scheduler 线程与事件开销
   - `autograd.grad` 与保存 SDPA 图的运行时开销
   - MoE 路由负载不均衡
   - NCCL 算法选择（ring/tree）与硬件拓扑

## 13. 小结

基于当前代码版本：

1. `Baseline` 与 `FluidMoE`（忽略重叠）在主项计算量与通信量上是等价的。
2. `AR` 通信量必须单独建模，且在两者间字节数相同（差异在是否被隐藏）。
3. 理论加速分析可先忽略 `n_msg * \tau` 与调度开销，先得到主项驱动的理想速度提升，再用实验解释偏差。

# ViT block00 SOMA 仿真报告

## NIR 与 residual 实现

本次使用 `full_blocks_until_silent_split_residual_norm` 的 block00：输入为
`block_input/metadata/output`，shape 由 NIR 读取为 `[30,1,64,128]`。只建立一份
8,192-neuron virtual input，直接导出 signed spike；没有 Patch Embedding、rate、Poisson
或其他编码。

residual 是通用的双 input weighted-identity fan-in，不按节点名进入 runtime。转换器读取
NIR `input1/input2`，逐一比对父节点 recorded output 后写入两条独立 connection。

| residual | input1 / wt1 | input2 / wt2 |
|---|---|---|
| residual_add1_IF | block_input / `1<<9=512` | proj_IF / `1<<7=128` |
| residual_add2_IF | residual_add1_IF / `1<<9=512` | fc2_IF / `1<<8=256` |

每一路 spike 正常经 SpikeQueue、XY NoC、identity synapse 到同一 destination Core 的
timestep buffer；转换器按 graph depth 给较短分支设置 delay（本例 5/0 与 3/0），从而在
同一 logical timestep 做一次 ST-BIF state transition。每个来源的 scalar accumulation 都
计入 synaptic update、packet、hop、congestion 和 identity energy。

norm 为独立逐 channel affine ST-BIF：`weight[c] << shifter[c]` 进入 membrane；replay
schema 的 bias 已移位，仅用于 `floor(multiplier/2)+bias` initial membrane preload，不会
重复移位或逐 timestep 加入。flush 自动由完整 NIR T、graph depth 与最后非零 input timestep
推导，Cx State 不会因 CSV 省略零 spike 而提前停止。

## QK/QKV 与 Cx State

QK/QKV 使用配置驱动的 `incremental_spike_matmul`。QK：
`S_Q(old)K_t^T + Q_tS_K(old)^T + Q_tK_t^T`；QKV：
`S_A(old)V_t + A_tS_V(old) + A_tV_t`。head、row、column、reduction、layout 从 NIR tensor
和 operand schema 推导，不写死当前 2/64 值。QK/QKV 各 physical Core 只维护本输出分区所需
的 local persistent shadow，当前 timestep operand 是清空的 sparse transient map；不会跨
Core 读取 tracer 或传 dense shadow。

普通 ST-BIF membrane、tracer、threshold 与 firing counter 都在本 Core 的 SoA Cx State。
ST-BIF leak=1；tracer 不额外计 SRAM 或 soma access。`state_start_timestep` 是 mapping
配置字段，按 graph depth 使每个 Core 恰在其第一帧输入对应 timestep 启动；启动后无 packet
的静默帧仍演化 persistent state。

## 结果与正确性

实际运行 `output/vit_block00/summary.json`：`completed=true`，136 physical Cores、34 tiles，
hardware latency 15,002,371,047 ps，energy 2,101,448,283.69 pJ，753,414 packets，
3,490,877 NoC hops，33,610,991 synaptic updates，23,396,352 attention updates。

`output/vit_block00/correctness.json` 对 NIR metadata/output 做逐 timestep、逐元素 signed
spike 对比，13 个计算节点全为零 mismatch：norm1、q/k/v、qk/qkv、proj、residual_add1、norm2、
fc1/fc2、residual_add2、norm1_next。Focused C++ test 还覆盖 +1/0/-1 QK/QKV 增量恒等式，以及
两条独立 scalar weight 的双输入 residual 累加。

主要普通 synapse 工作量为 fc1 的 17,599,488 updates；QK/QKV 分别为 11,010,048 与
12,386,304 attention updates；两个 residual 分别为 75,734 与 79,195 updates。

## 增量修改与 VGG 回归

修改仅限配置、NIR runtime 资产转换、mapping 的通用 `state_start_timestep`、以及全层
correctness 工具；双输入 residual 复用已有 multi-connection fan-in/identity synapse，不改变
全局 SpikeQueue、Core、NoC 或 VGG/ResNet 输入注入语义。相关入口为
`configs/vit_loihi_like.yaml`、`tools/prepare_vit_block00_assets.py`、
`tools/compare_vit_block00_reference.py`。

最终 VGG 256-step 使用 `output/vgg16_split_residual_regression_256/` 对比
`output/vgg16_truenorth_regression_256/`：所有既有 summary、timestep CSV、layer CSV 硬件共同
字段完全一致：hardware latency 638,970,424,703 ps，energy 163,709,593,412 pJ，packets
177,386,828，NoC hops 2,210,196,692，synaptic updates 5,385,396,144；新增 attention 字段为
0。本次 host 为 429.776 s，基线保存值为 339.173 s；这是单线程 wall-clock 的本机负载波动，
硬件模拟字段没有变化。

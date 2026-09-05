# Spikingformer ImageNet SOMA 仿真状态报告

## 实际运行

已使用官方 checkpoint 和项目内 dog 样例完成一次端到端 SOMA 运行：
`output/spikingformer_imagenet/summary.json` 的 `completed` 为 `true`。
输入是预处理后 `[T,C,H,W]` 的 binary spike CSV，未使用 rate/Poisson 编码；CPU golden 使用官方模型的 torch backend。

## SSA 与 Core-local 状态

SSA 由配置驱动的 `timestep_spike_attention` 实现。Q/K/V 只在当前 logical timestep 形成稀疏 transient，目标 Core 保存本地 KV，按配置推导 head、token、reduction 和 partition，不展开 neuron-to-neuron attention connectivity；统计中分别记录 Q、KV 及总 attention updates。

残差和卷积/BN 等多值路径使用 Loihi2-like 的 Core-local multi-valued state buffer（可编程 Cx State 语义）。每个 neuron 的整数/定点状态留在本 Core，沿 `local_state_buffer` 传给下游，不生成 spike、不经 NoC、跨 timestep 保留；普通 LIF neuron 的 membrane/tracer 仍在 Core-local neuron state 中。

## 规模与统计

本次 mapping 使用 35,280 个 physical cores、8,820 个 tiles。最终统计如下：

| 指标 | 数值 |
|---|---:|
| hardware latency | 5,477,004,500,498 ps |
| host latency | 6,046.93 s |
| energy | 5.8595540018e13 pJ |
| packets | 546,053,035 |
| NoC hops | 40,291,490,429 |
| synaptic updates | 52,104,600,796 |
| attention updates | 1,608,303,771,648 |
| KV / Q updates | 1,597,436,854,272 / 10,866,917,376 |

开销最大的 layer 是各 block 的 attention（单层约 298–337 s host latency、201,037,971,456 attention updates）；其次是 MLP1（约 3.4–4.1e9 synaptic updates）。

## 正确性边界

仓库中的 `soma-focused-tests` 已覆盖带有 +1/0/-1 spike 的 SSA 增量恒等式；构建和 CTest 通过。当前完整运行的 `output_scores` 仍为零向量，官方 PyTorch golden prediction 为 258，因此最终 readout 及逐节点 reference comparator 尚未达到 zero mismatch，不能宣称模型输出已对齐。VGG16 历史回归的 hardware metrics 保持不变（hardware latency 638,970,424,703 ps、packets 4,770,334、NoC hops 与 synaptic updates 一致）；本报告记录为回归基线，尚需在最终 comparator/readout 修复后再复跑确认。


# ViT block00--05 端到端 NIR 回放结果

运行配置：[configs/vit_loihi_like_blocks00_05.yaml](../configs/vit_loihi_like_blocks00_05.yaml)。
实际命令：

```bash
/home/ydy/miniconda3/envs/sim_snn/bin/python tools/run_vit_blocks.py \
  --config configs/vit_loihi_like_blocks00_05.yaml
```

六个 NIR 均为 `completed=true`，所有计算节点逐 timestep、逐元素与 NIR reference 完全一致。相邻五个边界也均为零 mismatch：前一 block 的 `residual_add2_IF` 输出等于后一 block 的 `block_input`。

最终模型输出是 block05 的 `residual_add2_IF`；其 245,760 个时空元素零 mismatch。

| block | physical cores / tiles | hardware latency (ps) | packets | host latency (s) | correctness |
|---|---:|---:|---:|---:|---|
| 00 | 136 / 34 | 15,002,371,047 | 753,414 | 3.020 | exact |
| 01 | 136 / 34 | 19,661,418,456 | 943,982 | 4.005 | exact |
| 02 | 136 / 34 | 21,093,151,741 | 1,027,106 | 4.776 | exact |
| 03 | 136 / 34 | 25,343,679,626 | 1,263,915 | 5.428 | exact |
| 04 | 136 / 34 | 27,526,134,032 | 1,410,904 | 6.594 | exact |
| 05 | 128 / 32 | 30,351,131,324 | 1,462,598 | 7.415 | exact |

累计串行硬件工作量为：138,977,886,226 ps（0.138978 s）、22,218,077,463.47 pJ（22.218 mJ）、6,861,919 packets、31,919,624 NoC hops、350,838,872 synaptic updates 和 300,220,416 attention updates；六次 simulator host latency 合计 31.238 s。

这里的累计量是六个 block 按 NIR 样本边界分别实际仿真后的串行和：每次运行都遵循 NIR 的 sample/Cx State reset 边界。它不是把六组 physical Core 同时塞入一个巨大 mapping 后测得的单次并行关键路径；单 block 的关键路径时延如表所示。完整机器可读结果见 [end_to_end_summary.json](../output/vit_blocks00_05/end_to_end_summary.json)。

## 本轮增量修正

block05 的 `q_IF` metadata 使用负 shifter `-3`。NIR 的实际整数语义是每个 destination 在一个 timestep 内先累加 `input × weight × 2^shifter`，然后 nearest-even 量化，再执行 ST-BIF；不能把负 shifter 预先整数右移到每条 signed weight。新增通用 `post_accumulation_rounding: nearest_even` mapping 字段，Core 在 soma phase 前量化其本 timestep buffer。该字段不依赖 node 名或具体 shape，默认 `none`，不会改变 VGG/ResNet 流程。

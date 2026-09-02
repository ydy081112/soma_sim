# Verified results

## Minimal spatial sample

- Queue drained: yes
- Processed packets: 5
- Output scores: `[0, 0]`（固定 2 timestep 时最后一批尚未到达 readout neuron phase）
- Prediction / expected: `0 / 0`
- Focused CTest: 1/1 passed

## VGG16 IF-SNN / CIFAR-10 sample 000 / 256 timesteps

- Queue drained: yes
- Input spikes: 219,843
- Emitted logical spikes: 4,770,334
- Hardware latency: 0.638970424703 s（含每步 1.8 us synchronization）
- SANA-FE hardware latency: 0.566443577764 s；SOMA 相对误差 `+12.8039%`
- 平均每 timestep hardware latency: SOMA `2495.978221 us`，SANA-FE `2212.670226 us`
- Dynamic event energy: SOMA `163.709593412 mJ`，SANA-FE `159.962050174 mJ`，相对误差 `+2.3428%`
- 平均每 timestep energy: SOMA `639.490599 uJ`，SANA-FE `624.851758 uJ`
- Host latency: 330.478 s on the current host；该 wall-clock 数值不参与 hardware latency
- Physical mapping: 279 cores / 70 tiles；5 个 AvgPool 均已融合，无独立 Pool stage
- Packets / NoC hops / synaptic updates: 177,386,828 / 2,210,196,692 / 5,385,396,144
- Spatial / Dense synaptic updates: 5,354,148,864 / 31,247,280；分别按 3.1 ns / 3.8 ns 计费
- Timestep 1 hardware latency: 11,732,800 ps
- SOMA output prediction / label: `3 / 3`
- Preserved SANA-FE reference prediction: `3`
- Output-score cosine similarity to SANA-FE reference: `0.9998356`

SOMA scores:

```text
[-289.3839, -1152.2512, -137.1532, 2556.1306, -227.8952,
  151.4852,    -1.8354, -185.4681, -419.7989, -294.2643]
```

SANA-FE reference scores:

```text
[-267.3911, -1042.6231, -96.1076, 2297.1132, -188.7913,
  149.9703,   -27.2136, -156.7256, -390.7198, -267.2784]
```

两者分类一致，score cosine 为 `0.9998356`，但不能把接近的 score 当作逐值相等。Pool-fused 图、physical placement 和紧凑 connectivity 已对齐；当前 packets/hops/updates 仍比 SANA-FE 高约 2.6%–3.0%，量化与 threshold 边界按本轮要求暂未调整，逐 timestep hardware latency 也仍受现有 NoC contention 模型差异影响。当前 latency 使用 YAML 驱动的 destination-router/per-Core input FIFO 回压；前 64 步总量与先前保存的 `Destination-router 4 queues` 候选精确一致。

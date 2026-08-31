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
- Emitted logical spikes: 5,176,445
- Hardware latency: 0.5678836707 s（含每步 1.8 us synchronization）
- SANA-FE hardware latency: 0.5664435778 s；SOMA 相对误差 `+0.2542%`
- Host latency: 206.031 s on the current host
- 已保存的 SANA-FE host simulation-call 合计为 2140.541 s；当前观测中 SOMA 少用 1934.510 s（`90.3748%`），即 `10.3894x` 快
- Peak RSS including compressed/uncompressed weights and packet queue: about 203 MiB
- Physical mapping: 310 cores / 78 tiles（当前 SOMA 图仍保留 5 个显式 AvgPool）
- Packets / NoC hops / synaptic updates: 146,488,599 / 1,813,376,089 / 3,569,567,135
- SOMA output prediction / label: `3 / 3`
- Preserved SANA-FE reference prediction: `3`
- Output-score cosine similarity to SANA-FE reference: `0.9996368`

SOMA scores:

```text
[-275.4394, -1113.5419, -144.1083, 2486.1082, -207.3201,
  177.1038,   -19.6900, -215.5919, -388.3068, -299.6349]
```

SANA-FE reference scores:

```text
[-267.3911, -1042.6231, -96.1076, 2297.1132, -188.7913,
  149.9703,   -27.2136, -156.7256, -390.7198, -267.2784]
```

两者分类一致，score cosine 为 `0.9996368`，但不能把接近的 score 当作逐值相等。当前 SOMA 图保留显式 AvgPool，而 SANA-FE benchmark 将 pooling 融合进后续 Conv/FC。Host 对比也不是严格同机 benchmark：SOMA 数据来自本机 i7-11700，SANA-FE 数据来自远程 Xeon Platinum 8352S，且 SANA-FE 使用 256 次 `chip.sim(1)` 与 `perf_trace=True`。因此 `10.3894x` 只是当前端到端观测差，不能单独归因于 CPU 或 simulator 实现。

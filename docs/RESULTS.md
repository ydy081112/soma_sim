# Verified results

## Minimal spatial sample

- Queue drained: yes
- Processed data spikes: 5
- Output scores: `[0, 0]`（固定 2 timestep 时最后一批尚未到达 readout neuron phase）
- Prediction / expected: `0 / 0`
- Focused CTest: 1/1 passed

## VGG16 IF-SNN / CIFAR-10 sample 000 / 256 timesteps

- Queue drained: yes
- Input spikes: 219,843
- Processed data spikes: 5,176,477
- Hardware latency: 0.2253739237 s（不含人为 timestep 间隔）
- Host latency: 10.4269 s on the current host
- Peak RSS including compressed/uncompressed weights: about 132 MiB
- SOMA output prediction / label: `3 / 3`
- Preserved SANA-FE reference prediction: `3`
- Output-score cosine similarity to SANA-FE reference: `0.9996414`

SOMA scores:

```text
[-275.0067, -1113.7877, -143.8189, 2485.6475, -206.9778,
  176.4586,   -20.1396, -215.2306, -388.0768, -299.4881]
```

SANA-FE reference scores:

```text
[-267.3911, -1042.6231, -96.1076, 2297.1132, -188.7913,
  149.9703,   -27.2136, -156.7256, -390.7198, -267.2784]
```

两者分类一致，但不能把接近的 score 当作 cycle-accurate 数值相等：SOMA 已按 timestep buffer 聚合后统一 neuron processing，但本版 VGG mapping 仍使用 aggregated partition。机器相关的 `host_latency` 也不是固定验收常数。

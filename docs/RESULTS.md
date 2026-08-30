# Verified results

## Minimal spatial sample

- Queue drained: yes
- Processed data spikes: 6
- Output scores: `[3, 0]`
- Prediction / expected: `0 / 0`
- Focused CTest: 1/1 passed

## VGG16 IF-SNN / CIFAR-10 sample 000 / 256 timesteps

- Queue drained: yes
- Input spikes: 219,843
- Processed data spikes: 5,336,573
- Hardware latency: 8.8170754285 s（不含人为 timestep 间隔）
- Host latency: 21.4050 s on the current host
- Peak RSS including compressed/uncompressed weights: about 132 MiB
- SOMA output prediction / label: `3 / 3`
- Preserved SANA-FE reference prediction: `3`
- Output-score cosine similarity to SANA-FE reference: `0.9988768`

SOMA scores:

```text
[-299.2880, -1223.5084, -138.8058, 2734.7698, -268.9496,
  111.0878,   -84.9267, -192.5496, -372.6299, -265.5867]
```

SANA-FE reference scores:

```text
[-267.3911, -1042.6231, -96.1076, 2297.1132, -188.7913,
  149.9703,   -27.2136, -156.7256, -390.7198, -267.2784]
```

两者分类一致，但不能把接近的 score 当作 cycle-accurate 数值相等：SOMA 在 timestep barrier 内对每枚到达 spike 直接完成 threshold/firing，参考转换在完整 timestep 聚合后 threshold；此外本版 VGG mapping 使用 aggregated partition。机器相关的 `host_latency` 也不是固定验收常数。

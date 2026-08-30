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
- Processed data spikes: 5,324,772
- Host latency: 21.275 s on the current host
- Peak RSS including compressed/uncompressed weights: about 142 MiB
- SOMA output prediction / label: `3 / 3`
- Preserved SANA-FE reference prediction: `3`
- Output-score cosine similarity to SANA-FE reference: `0.9990186`

SOMA scores:

```text
[-303.4142, -1223.3387, -143.2501, 2754.5874, -282.1200,
  132.6397,   -90.8800, -181.6844, -383.5562, -279.3705]
```

SANA-FE reference scores:

```text
[-267.3911, -1042.6231, -96.1076, 2297.1132, -188.7913,
  149.9703,   -27.2136, -156.7256, -390.7198, -267.2784]
```

两者分类一致，但不能把接近的 score 当作 cycle-accurate 数值相等：SOMA MVP 在每枚到达 spike 后 threshold，参考转换在完整 timestep 聚合后 threshold；此外本版 VGG mapping 使用 aggregated partition。机器相关的 `host_latency` 也不是固定验收常数。

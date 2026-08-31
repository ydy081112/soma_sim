# VGG16 CIFAR-10 benchmark

本项目不训练模型。`tools/prepare_vgg16_assets.py` 接收已有的 SANA-FE export：

- `snn_parameters.npz`：13 个 Conv、2 个 IF FC 和一个 integrate-only readout；
- `sample_000.npz`：CIFAR-10 单张图片的 256-step、6-channel ON/OFF rate coding；
- 已知 sample label/prediction 均为 `3`。

准备资产：

```bash
python3 tools/prepare_vgg16_assets.py \
  --source-parameters input/vgg16_source_parameters.npz \
  --sample input/cifar10_sample_000.npz \
  --timesteps 256
```

脚本把 `[Kh,Kw,Cin,Cout]` 重排为 `[Cin,Kh,Kw,Cout]`，为每层 deduplicate spatial plan，并生成 `compiler/mapping_output/vgg16_mapping.yaml`。每层按最多 1024 neurons 连续映射到 physical Cores，空间层采用与 SANA-FE 相同的 channel-major 物理顺序；一个 firing 按 destination Core 集合 packetize，但仍通过 Spatial Pattern 在 packet 到达时生成 local updates，不展开全量 synapse。

运行：

```bash
./build/soma-sim \
  --mapping compiler/mapping_output/vgg16_mapping.yaml \
  --weights input/vgg16_weights.npz \
  --input input/vgg16_input_spike.csv \
  --output output_vgg16
```

完整 256-step 是 benchmark，不进入默认 CTest。可先加 `--max-events 100000` 检查吞吐与内存，但这种输出会明确标记 `completed: false`，不能当成分类结果。

结果中的 `hardware_latency` 是目标硬件时间轴上的模拟时延；`host_latency` 是当前主机执行模拟器的 wall-clock 耗时。VGG 输入的逻辑 timestep 从 1 到 256，CSV 中 `generated_time/current_time` 全部为 0；每步先处理上一 timestep buffer，再排空本步 Data/NoC/accumulation。sample 000 的 timestep 1 没有外部 spike；满载 physical Core 的 soma processing 加同步 barrier 为 `11,732,800 ps`，不会人为占用 1 秒。

当前同步语义会先聚合同一 timestep 内针对同一 neuron 的所有 synaptic input，下一步只做一次 state update 和至多一次 firing。固定运行 256 timestep 时，结果表示 timestep 256 neuron phase 后的状态；该步 Data 写入的 buffer 将留给未来 timestep，不自动增加额外 flush step。`reference/` 中保留了 SANA-FE/Python 的已知结果用于比较。

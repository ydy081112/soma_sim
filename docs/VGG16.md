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

脚本把 `[Kh,Kw,Cin,Cout]` 重排为 `[Cin,Kh,Kw,Cout]`，为每层 deduplicate spatial plan，并生成 `compiler/mapping_output/vgg16_mapping.yaml`。大于 1024 neuron 的 VGG layer 在本 MVP 中标成 `partition: aggregated`，同时记录 `aggregate_core_count`；这用于快速架构级运行，不等价于编译栈未来生成的逐 Core multicast mapping。

运行：

```bash
./build/soma-sim \
  --mapping compiler/mapping_output/vgg16_mapping.yaml \
  --weights input/vgg16_weights.npz \
  --input input/vgg16_input_spike.csv \
  --output output_vgg16
```

完整 256-step 是 benchmark，不进入默认 CTest。可先加 `--max-events 100000` 检查吞吐与内存，但这种输出会明确标记 `completed: false`，不能当成分类结果。

结果中的 `hardware_latency` 是目标硬件时间轴上的模拟时延；`host_latency` 是当前主机执行模拟器的 wall-clock 耗时。VGG 输入默认用较大的 timestep period 隔离不同逻辑 step，因此 `hardware_end_time_ps` 是 timestamp，不能当作逐 timestep latency；跨实现比较前必须采用一致的 timestep latency 聚合口径。

当前事件语义在每个到达 spike 后触发 threshold；每个有输入的逻辑 timestep 会为带 bias 的层加入一个 `Bias` 事件。它仍是异步 spike-oriented 路径，与按 timestep 聚合完整卷积后只 threshold 一次的转换参考并非逐值等价。`reference/` 中保留了 SANA-FE/Python 的已知结果用于后续逐层校准。

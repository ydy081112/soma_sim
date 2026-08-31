# SOMA-Sim MVP

SOMA-Sim 是纯 C++17 的单线程、spike-oriented SNN architecture simulator。它用一个全局稳定优先队列逐枚处理 destination-Core packet，根据 mapping 的 physical Core/Tile 放置生成确定性 XY route，并用紧凑的 router output/link 可用时间表计算 NoC 拥塞；连接和神经元状态仍保持紧凑表示，不展开 neuron-to-neuron connection，也不创建 per-neuron C++ 对象。

## 构建与最小样例

```bash
python3 tools/generate_minimal_assets.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/soma-sim
```

结果写入 `output/summary.json`、`output/layer_metrics.csv` 和 `output/timestep_metrics.csv`。summary 额外报告 physical Core/Tile、packets、NoC hops、synaptic updates 和 soma/synapse/NoC/synchronization breakdown；输入 CSV 中的 `src_router/dst_router/route` 只用于 debug。

时间统计统一使用以下术语：

- `hardware_latency`：被模拟目标硬件的时延，内部用整数 ps，输出同时提供 ps/s；
- `host_latency`：主机实际执行 SOMA-Sim 所花的 wall-clock 时间，单位为 s；
- `generated_time/current_time/hardware_end_time`：硬件时间轴上的时间戳，不是 latency。

`summary.json` 使用 `hardware_latency_ps`、`hardware_latency_s`、`host_latency_s` 和 `host_processed_spikes_per_sec`；逐层/逐 timestep CSV 的主机耗时列统一为 `host_latency_s`。

常用参数：

```bash
./build/soma-sim \
  --hardware arch/hardware.yaml \
  --mapping compiler/mapping_output/mapping.yaml \
  --weights input/weights.npz \
  --input input/input_spike.csv \
  --output output
```

`--max-events N` 可用于大模型的受控 profiling；默认 `0` 会排空整个队列。

## 输入约定

- `hardware.yaml` 的 `hardware_latency` 支持 `ps/ns/us/ms/s`，加载时一次性转换成整数 ps。
- `mapping.yaml` 给出 layer 的起始 PE/Core、physical neuron order 和分区数；每个 packet 的静态 XY route 由 source/destination physical Core 所属 Tile 决定。
- `weights.npz` 使用无 pickle 的 NumPy 数组。Conv 权重为 `[Cin,Kh,Kw,Cout]`，Linear 为 `[Cin,Cout]`，并带 `plan_pattern_id`、`plan_dst_base`、`pattern_ptr`、`pattern_dst_offset`、`pattern_weight_offset`。
- 逻辑 neuron id 使用 spatial-major：`(y * W + x) * C + channel`；mapping 可用 `physical_neuron_order: channel_major` 指定 SANA-FE 风格的物理连续分区顺序。
- `hardware.yaml` 的 `architecture.execution_mode` 控制执行语义；当前 Loihi-style 配置使用 `timestep_synchronization`。
- 同步模式按“neuron processing → Data/NoC/synaptic accumulation → barrier”执行；一个 firing 按 destination physical Core 拆成 packet，Data 只写下一 timestep buffer，global queue 中只有真实 Data packet。
- soma timing 分为每个 mapped neuron 的 `soma_access`，以及仅对实际 state update 收取的 `soma_update`。
- `input_spike.csv` 的逻辑 `timestep` 从 1 开始；同步模式下 `generated_time/current_time` 均写 0，实际硬件时间由 simulator 在逐 timestep 注入时生成。

用原始 CIFAR-10 pickle 或项目提供的 sample NPZ 生成 rate-coded 输入：

```bash
python3 tools/encode_cifar10_rate.py \
  --cifar-batch /path/to/cifar-10-batches-py/test_batch \
  --sample-index 0 --timesteps 16 --mode deterministic \
  --output input/cifar10_input_spike.csv
```

完整 VGG16 资产准备和运行见 [docs/VGG16.md](docs/VGG16.md)，本次已验证结果见 [docs/RESULTS.md](docs/RESULTS.md)。设计边界见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

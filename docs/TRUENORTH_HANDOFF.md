# TrueNorth / NeMo VGG16 适配交接

更新时间：2026-09-03

## 当前状态

当前没有运行中的资产生成、构建或仿真进程；尚未提交 Git，也没有修改远程服务器。工作树包含此前从服务器同步的用户改动和本轮 TrueNorth 增量，禁止整体回退或覆盖。

目标是以服务器 `/home/dingyang_yu/comparison/NeMo` 和 `/home/dingyang_yu/comparison/nemo_vgg16_snn_cifar10/NOTES.md` 为 ground truth，在 SOMA-Sim 中通过通用 YAML 能力复刻其 TrueNorth VGG16 mapping 与逐 timestep firing 行为，同时保证原 Loihi VGG16 的功能、hardware latency、energy 和 host 热路径不变。

远程服务器只用于只读核验；本轮未运行 GPU，未修改远程文件。

## 已确认的 ground truth

- NeMo 模型：19,860 physical cores、2,944,798 个配置 neuron、241,313,977 个 enabled connections、2,944,788 条 neuron routes。
- 每个 Core 为 256 axons、256 neuron slots、256×256 binary crossbar、4 axon types 和每 neuron 4 个整数权重。
- 输出位于 core 19859、local neuron 0–9。
- SQLite 输入最早 timestamp 为 2；16-timestep ground truth 只覆盖 timestep 1–16。
- NeMo `--svs` 在 commit 时写 `tw_now + 1`，统计按 `floor(timestamp)` 分组。
- NeMo 16-step firing：

| Timestep | Firing events | Fired neurons |
|---:|---:|---:|
| 1–3 | 0 | 0 |
| 4 | 170,732 | 170,732 |
| 5 | 190,614 | 190,614 |
| 6 | 383,977 | 383,977 |
| 7 | 428,457 | 428,457 |
| 8 | 996,150 | 996,150 |
| 9 | 986,686 | 986,686 |
| 10 | 1,006,214 | 1,006,214 |
| 11 | 1,026,636 | 1,026,636 |
| 12 | 1,026,831 | 1,026,831 |
| 13 | 1,043,005 | 1,043,005 |
| 14 | 1,032,226 | 1,032,226 |
| 15 | 1,200,634 | 1,200,634 |
| 16 | 1,234,723 | 1,234,723 |

NeMo 完整 256-timestep ground truth 尚未完成，不能伪造或用 16-step 数据外推。

## 当前本地资产

大型源资产和生成资产均在 `.gitignore` 中：

- `input/truenorth_vgg16_nemo_model.nfg1`
  - SHA-256：`f6ab0bc85557630b77915818678995baae562c4b77396e02cc65b8c7a741ecb6`
- `input/truenorth_vgg16_nemo_spikes.sqlite`
  - SHA-256：`7548190e1e7f7f3a81245169e3dfefe4fec1fdb491a29170e2e9acd2bdab97ed`
- `input/truenorth_vgg16_weights.npz`，约 273 MiB
- `input/truenorth_vgg16_input_spike_t16.csv`，约 3 MiB、105,980 行 input events

资产生成工具为 `tools/prepare_truenorth_vgg16_assets.py`；它使用 Spatial/crossbar 紧凑表示，没有展开 neuron-to-neuron edge。

## 已完成的代码增量

- `HardwareConfig`：增加 max axons、Core/Tile/Chip placement、crossbar latency、event-activated neuron update、membrane bounds、positive saturation、firing report offset，以及 crossbar 行为配置。
- `MappingConfig`：增加 Crossbar layer/connection、direct input 和 readout neuron range。
- NPZ/weight runtime：增加 i16/u8/u16/u64 数组加载及 TrueNorth threshold、active slot、axon type、4 weights、bit-packed crossbar rows、route destination core/axon。
- `SynapseEngine`：按 axon-major bit-packed row 做 crossbar accumulation，并按紧凑 route 数组 packetize。
- `Core`/`SomaState`：支持 per-neuron threshold、fire count、membrane clamp、event-activated catch-up 和 crossbar packet 激活语义。
- `Simulator`：支持 direct input、crossbar destination axon、readout firing count 和 `firing_metrics.csv`。
- `TileLayout`：增加 chip 字段及 cores-per-tile/cores-per-chip placement。
- 新增 `arch/truenorth.yaml` 和 `compiler/mapping_output/truenorth_vgg16_mapping.yaml`。

输入 route 没有复制进 CSV。CSV 的 `src_neuron` 是输入记录索引，`tn_input_route_destination_partition/axon` 资产才是目标 route 的唯一真值，符合项目约束。

## 最近验证结果

构建与单测通过：

```bash
cmake --build build -j8
ctest --test-dir build --output-on-failure
```

最近一次 16-step 命令：

```bash
./build/soma-sim \
  --hardware arch/truenorth.yaml \
  --mapping compiler/mapping_output/truenorth_vgg16_mapping.yaml \
  --weights input/truenorth_vgg16_weights.npz \
  --input input/truenorth_vgg16_input_spike_t16.csv \
  --output output/truenorth_vgg16_t16_inactive
```

结果：

- host latency：3.861852279 s（`/usr/bin/time` elapsed 5.72 s，peak RSS 709,708 KiB）
- hardware latency：14,614,000 ps；TrueNorth YAML 当前全部硬件 service latency 为 0，因此该值主要受既有统计/free-time 路径影响，尚不能当作 TrueNorth ground truth。
- packets：671,591
- synaptic updates：37,890,638
- total neuron firings：7,624,741
- energy：0；当前参考 TrueNorth YAML 没有有效动态能耗参数，未用调参补值。
- output core 仍未 firing，16-step prediction=0 不具有分类意义，NeMo 的 16-step output 同样为 0 firing。
- 可直接用于绘图/核对的前 16 timestep 表已写入 `output/truenorth_vgg16_t16_inactive/nemo_firing_comparison.csv`；每 timestep 的 firing event 与 fired neuron 在这两个实现当前统计口径中相同。

当前 SOMA firing：

| Timestep | SOMA | NeMo |
|---:|---:|---:|
| 1–3 | 0 | 0 |
| 4 | 150,100 | 170,732 |
| 5 | 175,960 | 190,614 |
| 6 | 198,642 | 383,977 |
| 7 | 230,454 | 428,457 |
| 8 | 327,605 | 996,150 |
| 9 | 605,650 | 986,686 |
| 10 | 779,276 | 1,006,214 |
| 11 | 705,645 | 1,026,636 |
| 12 | 663,133 | 1,026,831 |
| 13 | 711,231 | 1,043,005 |
| 14 | 794,307 | 1,032,226 |
| 15 | 829,638 | 1,200,634 |
| 16 | 751,307 | 1,234,723 |

`firing_metrics.csv` 还会出现 timestep 17，因为 timestep 16 的内部 processing 产生了 report offset 后的 firing；与 NeMo `--end=16` 比较时只应取 1–16，并继续核对仿真结束边界。

## 已定位但尚未解决的语义

1. NeMo `TNIntegrate` 中实际代码是：

   ```c
   weight = synapticWeight[axonTypes[synapseID]] &&
            synapticConnectivity[synapseID];
   ```

   所以任意非零配置权重在有效连接上贡献 `+1`。本地已通过通用配置 `crossbar_weight_mode: nonzero_binary` 表达，默认仍为 `signed`，不影响 Loihi。

2. NeMo 收到一个 `AXON_OUT` 后，会给该 Core 的全部 256 neuron slots 生成 `SYNAPSE_OUT`。即使 crossbar bit 为 0，也会安排 neuron heartbeat。本地已增加：

   ```yaml
   crossbar_packet_activates_all_neurons: true
   ```

3. 更关键的是 NeMo 的 `TN_forward_event` 没有跳过 `isActiveNeuron=false` 的 slots。nfg1 未配置的 slot 保持 calloc 零状态，因此 threshold=0；当 Core 收到广播时，这些 inactive slots 也会 firing。远端 t4 原始 CSV 证据：170,732 次 firing 恰好来自 781 个 input target cores，其中 local neuron `<64` 的配置 neurons 为 20,780 次，local neuron `>=64` 的 inactive slots 为 149,952 次。

   本地已将其做成默认关闭的通用兼容项：

   ```yaml
   process_inactive_neurons_on_crossbar_event: true
   ```

   资产生成器也把 inactive threshold 改为 0。启用后 SOMA t4=150,100，已接近 NeMo 的 149,952 个 inactive-slot firings，但仍缺 20,632 次 active-neuron firing。

4. 已定位 active neuron 的 t4 首次差异，并作为通用硬件配置实现。`core.neuron.threshold_compare_mode` 默认为 `signed`；TrueNorth YAML 设为 `unsigned_promotion`，将 int32 membrane 位模式提升成 uint32 再做现有 `greater_equal` 判断。`919:63`、`988:63`、`521:63` 分别复现 reset `-312/-316/-320`。完整 t1–16 firing MAPE 从 `33.39%` 降至 `5.68%`，但 t5 后 promotion firing 的 recurrent propagation 仍带来 false positives；精确 set comparison 与前后表见 `output/truenorth_vgg16_unsigned_promotion_t16/firing_comparison.md`。未按 core/local 特判，未改 latency/threshold 常数。

5. 当前 `event_activated_catch_up` 用 `configured_bias * elapsed` 近似 heartbeat catch-up，但 NeMo 的 leak 公式依赖 `sigma_l`、`lambda`、epsilon、旧膜电位符号和 `lastLeakTime`。TrueNorth 需要独立、配置选择的 integer leak/reset 语义，不能污染现有 Loihi `voltage *= leak` 路径，也不能按芯片名分支。

6. TrueNorth mapping 当前用 64×320 mesh 容纳 19,860 Core，并配置 `cores_per_chip=4096`。NeMo 是五个连续 Core-ID chip；当前没有独立片间 NoC timing。因为 YAML latency 全为 0，这不影响本轮 firing 对比，但后续若声明 hardware latency，需要先补可配置 chip-aware routing/latency。

## 256-step SOMA 运行

完整输入已生成并在本地完成一次 SOMA 256-step 运行：

```bash
/usr/bin/time -f 'elapsed=%e maxrss_kb=%M' ./build/soma-sim \
  --hardware arch/truenorth.yaml \
  --mapping compiler/mapping_output/truenorth_vgg16_mapping.yaml \
  --weights input/truenorth_vgg16_weights.npz \
  --input input/truenorth_vgg16_input_spike.csv \
  --output output/truenorth_vgg16_256
```

- completed=true，55,405,450 packets，2,433,393,721 NoC hops，6,161,551,564 synaptic updates，430,263,354 firings。
- internal host latency `210.991057592 s`；`/usr/bin/time` elapsed `212.01 s`，peak RSS `720,092 KiB`。
- hardware latency `587,413,000 ps`，但 TrueNorth YAML 的 service latency/energy 均为零，当前值只可作为本实现的调度统计，不能和 NeMo tick 或真实 TrueNorth latency 对齐。
- energy 全部为 0；prediction=0、expected=3。NeMo 尚无完整 256-tick firing ground truth，因此不能给出完整逐 tick 功能比较。

## Loihi 回归

本轮增量后已重跑原 Loihi VGG16 256-step：

```bash
./build/soma-sim \
  --hardware arch/hardware.yaml \
  --mapping compiler/mapping_output/vgg16_mapping.yaml \
  --weights input/vgg16_weights.npz \
  --input input/vgg16_input_spike.csv \
  --output output/vgg16_truenorth_regression_256
```

- prediction=3，177,386,828 packets，5,385,396,144 updates，hardware latency `638,970,424,703 ps`。
- 相对 `output/vgg16_destination_router_fifo_256/`，`summary.json` 中所有原有非 host 字段一致；新增 `total_neuron_firings=4,550,491` 仅是新统计字段，旧输出未记录该字段。
- 去除 `host_latency_s` 和 `host_processed_spikes_per_sec` 后，`timestep_metrics.csv` 和 `layer_metrics.csv` 逐字段 `diff` 均为空。此次 host 为 `339.173 s`，仅代表 wall-clock 波动，不能与历史值逐次比较。

## 建议下一步

1. 后续若需进一步收敛 t5–t16，应审计 promotion 后 recurrent firing 的 event/tick ordering、`fireTimingCheck` 及 NeMo 对 active/inactive state commit 的细节；不得通过缩放 threshold/latency 或 neuron-id 拟合来压低剩余 false positives。
2. `statistics.firing_trace` 默认关闭，仅用于小步集合诊断；启用时写 `firing_trace.csv`，记录 report timestep、physical Core 和 local neuron，不影响默认 Loihi 热路径。
5. 只有服务器出现有效的 NeMo 256-step ground truth 后，才做完整 firing 表比较；SOMA 已有 256-step latency/energy 输出，但必须明确 NeMo firing 无可比数据。

## 注意事项

- 不要删除或回退当前工作树中已有注释和用户/服务器同步改动。
- `compiler/docs/` 当前为未跟踪目录，来源不属于本轮 TrueNorth 工作，交付前先审计，不要擅自加入或删除。
- 不要提交大型 nfg1、SQLite、NPZ、CSV 或 output。
- 恢复工作前重新完整阅读 `PROMPT.md`、`docs/background.md`、`docs/NOTES.md`，并先检查 `git status --short`。
- 远程只读；如需 GPU，必须先 `nvidia-smi`，且不得影响其他用户进程。

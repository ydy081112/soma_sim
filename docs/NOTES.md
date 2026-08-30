# 开发记录

## [进行中]

- 无。

## [已完成]

### 2026-08-31：timestep synchronization 注入与时间推进

- 在 `hardware.yaml` / `HardwareConfig` 中增加 `execution_mode: timestep_synchronization`，当前 Loihi-style 配置显式启用；当前版本拒绝未实现的其他 mode，但没有加入占位分支。
- input spike 的逻辑 timestep 改为从 1 开始，生成器和现有 CSV 的 `generated_time/current_time` 均为 0；删除 1 秒的人工 timestep period。
- simulator 只注入当前 timestep，完整排空其 Data/Bias/SomaDrain、NoC 和内部派生事件后，再从实际硬件完成时刻注入下一步；派生 spike 始终继承原逻辑 timestep。global queue、Core/NoC resource-free-time 和模块边界保持不变。
- 队列在同步模式按 `(timestep, generated_time, sequence_id)` 排序；默认构造仍保留硬件时间优先语义，便于后续无 timestep barrier 的架构扩展。
- 验证：Python 工具语法检查、Release 构建、CTest 1/1 和最小样例均通过；VGG16 256-step 全量队列排空，处理 5,336,620 个 data spike，硬件时延 19.8095648631 s，预测/标签/参考均为 3，score cosine 为 0.9989644。
- VGG sample 的 timestep 1 没有 input spike，但仍完成 Bias/神经元活动，`hardware_end_time_ps=635699200`；统计从 timestep 1 开始且不再生成 timestep 0 行。本节取代旧 benchmark 中依赖 `t * 1e12 ps` 并事后扣除 idle 的时间口径。

### 2026-08-30：关键实现注释补充

- 保留已有注释，并为 YAML/NPZ 解析、mapping、template 热路径、NoC 资源竞争、Core/Soma/fake spike、事件调度和统计边界补充 1–2 句中文说明。
- `cmake --build build -j2` 与 `ctest --test-dir build --output-on-failure` 均通过。

### 2026-08-30：仓库工作流初始化

- 在 `AGENTS.md` 增加可重复执行的协作、安全和验证流程。
- 建立本文件，后续每个阶段持续记录验证结果与限制。

### 2026-08-30：纯 C++17 simulator MVP

- 完成 CMake 工程和分文件实现：YAML 配置、NPZ/NPY、mapping、Spatial Pattern Template、source-major Conv/Linear、全局稳定 SpikeQueue。
- 完成紧凑 router output/link 资源表、方向 hop hardware latency/energy、req/data/ack/credit 驱动的 link mode、PE/Core input/SRAM/synapse/soma 时序。
- neuron state 使用 SoA；threshold 候选由 `SomaDrain` fake spike 每次发射一个，不做逐 tick 全状态扫描。带 bias 的层每个逻辑 timestep 使用一个 Bias 事件。
- 完成 JSON summary、逐层 CSV、逐 timestep CSV，包括 hardware latency、host throughput、五项 breakdown、组件 energy、scores/expected output。
- 完成最小 hardware/mapping/input 示例、CIFAR-10 rate encoder、最小权重生成器及 VGG connectivity/asset 工具。

### 2026-08-30：远程资产复用与 VGG16 benchmark

- 远程服务器 `58.220.114.154:7023` 密码连接正常；仅读取并复制用户目录内已有参数/sample/reference，没有运行 GPU 任务。
- 将 13 Conv + 5 AvgPool + 2 IF FC + readout 转换为 54.3 MiB source-major/template NPZ；输入为 CIFAR-10 sample 000 的 219,843 个真实 spike。
- 完整 256 timestep 队列排空：5,324,772 个 data spike，host latency 21.275 s，peak RSS 约 142 MiB，预测/标签/参考均为 3。
- SOMA scores 与保存的 SANA-FE scores 余弦相似度为 0.9990186；详细数值见 `docs/RESULTS.md`。
- 已知边界：大层使用 aggregated partition；异步逐 spike threshold 与参考的逐 timestep 聚合 threshold 不保证逐值相等。

### 2026-08-30：验证

- `cmake --build build -j2`：通过，无编译警告。
- `ctest --test-dir build --output-on-failure`：1/1 通过，覆盖时间换算、稳定队列、静态 route/争用、模板寻址和 fake-spike 排空。
- `./build/soma-sim`：最小样例排空，prediction/expected = 0/0。
- 四个项目工具与 connectivity compiler 均通过 `python3 -m py_compile`。

### 2026-08-30：VGG16 spike 统计口径核对

- 本地 `output_vgg16/summary.json` 和逐层/逐 timestep CSV 一致：处理 5,324,772 个 data spike、发射 5,324,772 个 spike、执行 3,766,357,619 次 destination/synaptic update；输入 trace 为 219,843 个 spike。
- 远程结果 `spikes=5,247,961,724` 来自 SANA-FE `PipelineUnit::process_synapse_input()` 的 `++spikes_processed`，按展开后的每条 synaptic connection 计数，不是 neuron firing/spike packet 数。同次 benchmark 另报 `neurons_fired=4,650,141`、`neurons_updated=71,263,427`、`packets_sent=172,905,295`。
- SANA-FE benchmark 把 5 个 AvgPool 分别融合进后续 Conv/FC，没有独立 pool firing；按逻辑 spike 口径补上 219,843 个外部输入后为 4,869,984。SOMA 去掉 437,336 个显式 pool firing 后，同口径为 4,887,436，相差 17,452（约 0.36%），来自已知的图结构和 threshold 调度语义差异。
- 验证方式：只读检查本地统计递增位置与结果求和；通过 SSH 只读检查远程 benchmark CSV、逐 timestep CSV 及 SANA-FE `src/pipeline.hpp`、`src/chip.cpp`、`src/pymodule.cpp`，未运行仿真或 GPU 工作，未改远程文件。
- 未覆盖边界：没有把两套网络改成完全相同的 pool 表示和 threshold 时序后重跑，因此 0.36% firing 差异已定位到模型语义范围，但未逐层归因到每一次 firing。

### 2026-08-30：VGG16 吞吐率与 Loihi hardware latency 误差核对

- Host 仿真吞吐按 `256 / host latency` 计算：SOMA-Sim 为 `256 / 21.275025049 = 12.032888 Timestep/s`；远程 SANA-FE 逐 step 的 `host_sim_call_s` 合计 `2140.541257870 s`，为 `0.11959592 Timestep/s`。现有记录下 SOMA-Sim 快 `100.6129x`。
- 两个 host latency 都排除了网络/权重构建阶段，但并非同机测量：本地为 i7-11700，远程为 Xeon Platinum 8352S；SANA-FE 还使用 256 次 `chip.sim(1)` 和 `perf_trace=True`，因此 `100.6129x` 是当前端到端 simulation-call 观测值，不是严格同机 microbenchmark。
- SANA-FE 的 256-step simulated Loihi hardware latency 为 `0.566443577764 s`。SOMA summary 原始值 `255.082552772 s` 不能直接比较，因为 VGG input CSV 默认把 timestep 放在 `t * 1e12 ps`，其中含人为 idle/时间原点；若直接误用，倍率为 `450.323x`、相对误差为 `+44932.30%`。
- 对本地逐 timestep 结果计算 `sum(hardware_end_time_ps - timestep * 1e12 ps)`，得到 255 个非空 step 的 active-span hardware latency `19.773924234 s`。相对 SANA-FE 的绝对误差为 `+19.207480656 s`，SOMA 为 `34.9089x`，以 SANA-FE 为分母的相对误差为 `+3390.8904%`。
- 大误差的主要已知来源：VGG mapping 虽记录 `aggregate_core_count`，当前 timing path 仍把每层当作一个聚合 Core；`Core::receive()` 使用 `synapse_hw_latency + updates * soma_update_hw_latency` 串行累计 destination updates，没有体现 SANA-FE 279 Core mapping 的并行度。两边 pool 表示、threshold 调度和 timestep synchronization 也尚未完全对齐。
- 验证方式：本地只读求和 `output_vgg16/summary.json` 与 `timestep_metrics.csv`；远程只读求和 `sanafe_vgg16_per_timestep.csv` 并交叉检查 benchmark/逐 timestep hardware latency，未重跑完整 benchmark、未使用 GPU、未改远程文件。

### 2026-08-30：hardware latency / host latency 术语统一

- 统一语义：`hardware_latency` 表示被模拟目标硬件的 duration；`host_latency` 表示主机执行模拟器的 wall-clock duration；事件的 `generated_time/current_time/hardware_end_time` 是硬件时间轴 timestamp，不称为 latency。
- C++ 内部按约定使用短名 `hw_*`，包括配置、resource reservation、Core/NoC result、breakdown 和 `SimulationResult::hw_latency_ps`；host 计时使用 `host_*`。对外 YAML/JSON/CSV 保留完整 `hardware_*`/`host_*`。
- `hardware.yaml` 已迁移为 `hardware_latency`、`directional_hardware_latency`、`busy_hardware_latency`、pipeline `*_hardware_latency` 和 core `hardware_latency` map；旧 YAML 键不再兼容。
- JSON schema：`total_simulation_time_{ps,s}` -> `hardware_latency_{ps,s}`，`cpu_simulation_time_s` -> `host_latency_s`，`cpu_processed_spikes_per_sec` -> `host_processed_spikes_per_sec`，`breakdown_cycles` -> `hardware_latency_breakdown_cycles`。
- CSV schema：`cpu_seconds` -> `host_latency_s`，`cpu_processed_spikes_per_sec` -> `host_processed_spikes_per_sec`，`simulation_end_ps` -> `hardware_end_time_ps`。同步更新 README、架构/VGG/结果文档、背景、PROMPT、测试和既有 `output/`、`output_vgg16/` 字段。
- 验证：`cmake --build build -j2` 通过；`ctest --test-dir build --output-on-failure` 1/1 通过，并新增 NoC/Core hardware latency schema 断言；`./build/soma-sim` 完成最小样例，终端同时输出 `hardware_latency_ps=440400` 与 `host_latency_s`，生成的 JSON/CSV 新字段通过检查。
- 未覆盖边界：没有重跑完整 VGG16 benchmark；`output_vgg16/` 只迁移字段名并保留原数值。任何仓库外 consumer 需要按上述映射同步迁移。

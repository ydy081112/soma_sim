# 开发记录

## [进行中]

- 2026-09-01：在新的 `/tmp` 干净副本中探索保持 spike-oriented/单次 packet 热路径的 destination backpressure：不使用第二遍 detailed scheduler、不创建内部 event、不保存全 timestep message trace、不修改 latency 参数。候选将 link propagation 与 occupancy 分离，并按 physical Core 建 Local ejection/input-buffer 状态（深度取 hardware config 的 16）；先跑 64/160-step 筛选，再完整跑 256-step，目标是所有 timestep 相对 SANA-FE 的最大绝对误差 `<20%`。正式仓库 simulator 保持不变，持续记录每个 ablation 的 host/hardware 指标。
  - 检查点 1：`/tmp/soma_sim_spike_backpressure.cG3EeD` 第一版用 per-Core 16-entry buffer，并在 Core processing start 释放 slot；Release build/CTest 1/1 通过，64-step host `66.7397 s`。结果与仅分离 propagation/occupancy 的 ablation 相同：总误差 `-22.976%`、MAPE `22.035%`、最大绝对误差 `56.477%`（timestep 4），说明 slot 在 processing start 释放时 Local buffer 没有在 critical path 形成足够反压。
  - 检查点 2：同一副本将 slot 改为 Core processing finish 释放，结果几乎不变；把 Local/最后 ingress link 一直占用到 Core finish 又过强（64-step 总误差 `+65.966%`、最大绝对误差 `262.596%`）。tile-shared depth-16 buffer 也几乎不在 critical path。说明不能把 destination processing delay 直接等价成某条物理 link 的完整占用期。
  - 检查点 3：`/tmp/soma_sim_credit_backpressure.sP2FFP` 测试 O(1) credit 状态。per-destination-Core depth-16 为总误差 `-22.129%`、MAPE `21.403%`、最大绝对误差 `48.349%`；聚合为每 Tile 一个 16-entry ingress 后为 `-18.776%` / `18.225%` / `43.525%`。把 route 上每个 Router 的 16 slots 保留到 Core finish 则总量高到 `+145.98%`，而逐 input-port slot 在下一 hop 释放又与弱模型相近。真实差异主要来自未满 buffer 时的在途密度，而不是单独的 buffer-full credit。
  - 检查点 4：`/tmp/soma_sim_online_backpressure.d5kkAy` 直接在原 global spike pop 中维护 SANA-style route density/min-heap。若 packet 到 destination processing start 即退出 NoC，总量低 `57.9%`；再传播 source blocking 时，由于 global queue 必须按 `(timestep, generated_time, sequence_id)` 而不是被阻塞后的 send time 排序，后处理的 packet 可能拥有更早 send time，在线密度状态失序并把总量推高到 `1.386 s`。因此不采用需要第二套 send-time event scheduler 的路径。
  - 检查点 5：`/tmp/soma_sim_flow_backpressure.H7RAew` 改用 compact per-flow 队列，不保存 timestep trace、不二次调度。按 destination Router 聚合且强制四个 Core 共用 FIFO 时，64-step 总误差 `+7.418%`、MAPE `9.873%`，但最大绝对误差 `53.152%`；将释放队列改为四个 physical Core 独立后，Release build/CTest 1/1 通过，host `78.986 s`，总误差改善为 `+1.087%`、MAPE `9.501%`，最大绝对误差仍为 t56 `+46.917%`，t2 为 `-42.377%`。这证明 destination-only 平均 service 仍会在个别高流量 step 过反馈，并缺少冷启动时真实的共享 route contention。
  - 检查点 6：只把每个 physical Core 的 `Port::Local` free time 延长到上一 packet 的 Core processing finish（不连带 directional link）后，64-step hardware 仍为 `99.1118402 ms`，与弱 pipeline 基线一致；这个语义没有进入最终 critical path。original asynchronous link 加 external virtual source Core 逐 packet 轮转同样几乎不改变曲线（总误差 `+20.446%`、最大绝对误差 `40.873%`），排除 CSV equal-time 顺序是 t2/t4 主因。
  - 检查点 7：将 flow 从 destination Router 细分到 `(destination Router, last ingress direction)` 后过弱：总误差 `-19.404%`、MAPE `19.700%`、最大绝对误差 `42.377%`。`/tmp/soma_sim_link_flow.c42EZG` 进一步按每条实际 route link 维护 processing-service min-heap，Release build/CTest 1/1 通过，但 64-step 总误差 `-20.360%`、最大绝对误差 `33.097%`，host 增至 `142.693 s`；精度和速度均不满足目标。
  - 检查点 8：已归档的严格 SANA detailed 候选仍是机制上唯一同时恢复 t2/t4 与整体趋势的模型：traffic 完全一致的 t2 为 `293.526256 us`，相对 SANA `293.471293 us` 仅 `+0.0187%`；但使用当前 SOMA firing traffic 的完整 256-step 最大绝对误差为 t147 `+26.5566%`。该 step 的 SOMA packets 比 SANA 多 `2.256%`、firing 多 `3.829%`、updates 多 `5.032%`，拥塞非线性放大后超过 20%。全程没有修改 latency 参数，也没有用 ground-truth packet count 做缩放或裁剪。
  - 下一步：核验正式 simulator SHA 与实验输出；在“暂不处理 neuron quantization/threshold 导致的 firing traffic 差异”约束下，记录 `<20%` 目标的可达性边界，不把不合格候选应用到正式代码。
- 2026-08-31：核对本机 i7-11700 与远程 Xeon Platinum 8352S 的 CPU、频率、缓存、NUMA 和仿真线程模型，评估 CPU 差异对 VGG16 `host_latency` / Timestep/s 的影响。假设沿用现有两套 256-step 结果，不重跑完整 benchmark；下一步采集两机硬件/负载信息并区分“CPU 性能差异”和“模拟器实现差异”。

## [已完成]

### 2026-09-02：正式合入 YAML 驱动的 Destination-router / per-Core FIFO

- `HardwareConfig::Core` 新增并加载 `input_fifo`、`fifo_per_core`、`fifo_num_per_core`、`fifo_depth_per_core`；当前配置启用每 Core 1 个、深度 16 的 FIFO。数量、深度和 Core 数均参与紧凑资源表，不在 simulator 热路径写死。
- NoC 保持 spike-oriented 单事件处理：destination flow 按 router 聚合 processing service，FIFO 释放队列和 Local ejection 按 physical Core 分离；不创建 FIFO event、packet trace 或第二套 scheduler。关闭 `input_fifo` 时保留原普通 traverse 路径。
- Release build 与 CTest 1/1 通过。完整 VGG16 256-step 处理 `177,386,828` packets，prediction/label=`3/3`，host latency `330.478495 s`，峰值 RSS `264,040 KiB`；前 64 步 hardware latency `130,074.773392 us`，与旧 Destination-router 4 queues 保存值在 CSV 精度下完全一致。
- 256-step SOMA/SANA hardware latency 总量为 `0.638970424703/0.566443577764 s`（`+12.803896%`），逐 timestep 平均为 `2495.978221/2212.670226 us`。Energy 总量为 `163.709593412/159.962050174 mJ`（`+2.342770%`），逐 timestep 平均为 `639.490599/624.851758 uJ`。
- 验证输出位于 `output/vgg16_destination_router_fifo_256/`。Energy 是两套配置的动态事件能耗总和，不含 leakage 时间积分；本轮 FIFO timing 不改变 packet/update/firing workload，因此 SOMA energy 与合入前一致。

### 2026-09-02：VGG16 256-timestep 总能耗复核

- 当前 `output_vgg16/summary.json` 的 SOMA-Sim 总能耗为 `163.709593412 mJ`；现有 SANA-FE 完整 benchmark 为 `159.9620501735714 mJ`。绝对差为 `+3.747543238429 mJ`，以 SANA-FE 为基准的相对误差为 `+2.342770197%`。
- 分组件 SOMA/SANA 与相对误差分别为：Synapse `129.249507456/126.274389202 mJ`（`+2.356074%`），Soma `5.487321942/5.478992567 mJ`（`+0.152024%`），Network `28.972764014/28.208668405 mJ`（`+2.708726%`），Dendrite 均为 0。
- 主要差异与功能 workload 一致：SOMA 相比 SANA 的 synaptic updates、firings、packets、hops 分别多 `2.6188%/2.5847%/2.5919%/2.9574%`；Soma 的实际 state updates 约少 `0.6022%`，抵消了 firing 增量，所以 Soma energy 最接近。
- 口径限制：这是两套模拟器使用当前各自 event/workload 和配置参数得到的动态事件能耗，不包含 leakage 的时间积分；SOMA 当前 synapse energy 使用统一 `24.0 pJ/update`，而 SANA-FE 对 Conv/Dense 分别使用 `24.0/35.5 pJ`，因此 `2.34%` 是端到端结果差，不是完全相同 workload 下的纯能耗模型误差。
- 验证：只读核对 SOMA summary、SANA-FE 256-step benchmark/逐 timestep CSV、两边硬件 energy 参数与 SOMA 统计累计代码，并用 `awk` 独立重算总量、分量及 workload 差；未重跑 benchmark、未使用 GPU、未修改远程文件。

### 2026-09-02：Destination-router 4 queues 逐 timestep 平均 latency/energy 误差

- 基于该方案已保存的全部 64-step CSV 与相同区间 SANA-FE 数据逐步对齐。Latency 平均值 SOMA/SANA=`2032.418334/2010.571169 us`，平均值误差 `+1.086615%`；逐步 signed error 均值 `+0.470293%`、MAPE `9.500865%`、最大绝对误差 `46.916938%`（t56）。
- SOMA timestep energy 按 packets、vertical hops、updates、updated neurons、firings 与配置能耗精确重建，64 步合计 `38.3801611003 mJ` 与 summary 完全一致。Energy 平均值 SOMA/SANA=`599.690017/584.772058 uJ`，平均值误差 `+2.551072%`；逐步 signed error 均值 `+2.494715%`、MAPE `2.552812%`、最大绝对误差 `6.128298%`（t3）。
- 结果已写入 `others/spike_oriented_tmp_comparison_20260902/destination_router_four_queues_timestep_average.csv`；统计只覆盖该旧 subvariant 实际保存的 64 steps，不外推为 256 steps。

### 2026-09-02：Spike-oriented `/tmp` timing 候选 latency/breakdown 复核

- 只复核仍保持 spike-oriented、单次 global event 热路径和紧凑状态的已有 64-step 候选；排除 detailed/exact 二次 scheduler、send-time 重排方案和 latency 参数拟合。未修改正式 simulator，也未触碰用户新换的绘图代码。
- SANA-FE 前 64 步为 `128.676555 ms`。累计 latency 最接近的是 destination-router four queues：`130.074773 ms` / `+1.087%`，MAPE `9.501%`，但最大逐步误差仍为 `46.917%`；最大逐步误差最低的 per-route-link flow 仍为 `33.097%`，且 host 增至 `142.693 s`。
- SANA serialized breakdown 为 Synapse/Soma/Network=`45.720800/2.486386/51.792814%`，所有 SOMA 候选均为 `45.737142/2.414736/51.848122%`，最大 component share 差仅 `0.072` 个百分点。原因是候选只改 waiting/backpressure，没有改变相同的 firing/packet/hop/update workload。
- 结论：合规候选都能保持几乎相同的 breakdown，但没有一个同时满足逐 timestep 最大误差 `<20%` 与速度优先。完整代表性结果和统计边界已固化在 `others/spike_oriented_tmp_comparison_20260902/`；旧 subvariant 当前无独立源码 snapshot，故没有伪造 256-step 重跑。

### 2026-09-02：Network 改为墨绿调浅灰绿

- Network 由薄荷浅绿 `#C7E9C0` 改为更沉稳、带墨绿蓝灰调的浅绿 `#A8CBB7`；浅蓝/浅橙及其他样式不变。PNG/PDF 已重新生成并目视检查，Windows PDF 已同步覆盖。

### 2026-09-02：Soma 百分比移到 stack 右侧

- 将四个 Soma 百分比从橙色色块内部移到对应 stack 的右侧，并与 segment 中心纵向对齐；位置由 `STACK_WIDTH` 自动计算。Synapse/Network 标签保持色块内部。PNG/PDF 已重新生成并目视确认无裁切或重叠，Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown 改用适配黑字的浅色配色

- stack 配色改为浅蓝 Synapse `#A6CEE3`、浅橙 Soma `#FDBF6F`、浅绿 Network `#B2DF8A`，保留黑色边界和黑色 normal-weight 百分比。PNG/PDF 已重新生成并目视确认对比度和类别区分清晰，Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown 百分比改为黑色

- stack 内 Synapse、Soma、Network 的百分比全部由白色改为黑色，保持 normal font weight；其他样式和数据不变。PNG/PDF 已重新生成并目视检查，Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown 百分比取消粗体

- stack 内 Synapse、Soma、Network 的白色百分比全部由 bold 改为 normal font weight；其他样式和数据不变。PNG/PDF 已重新生成并目视检查，Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown stack 再缩窄 1/5

- 保持画布、字号和布局不变，将 stack width 从 `0.675` 乘 `0.8` 调整为 `0.54`。PNG/PDF 已重新生成并目视确认标签完整；Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown 图高度再缩短 1/5

- 保持宽度、stack、字号和布局不变，将画布高度从 `2.80 in` 乘 `0.8` 调整为 `2.24 in`；PNG 高度变为 1394 px。PNG/PDF 已重新生成并目视检查，Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown stack 缩窄 1/4

- 保持 `5.20x2.80 in` 画布、字号和布局不变，将 stack width 从 `0.90` 乘 `0.75` 调整为 `0.675`。PNG/PDF 重新生成并目视确认标签完整；Windows PDF 已同步覆盖。

### 2026-09-02：Breakdown 图字号缩小至 2/3

- 保持 `5.20x2.80 in` 画布及 stack geometry 不变，将所有字号统一乘 `2/3`：百分比 `15→10 pt`，轴标题 `17→11.33 pt`，刻度/图例 `16→10.67 pt`，全局 `16.4→10.93 pt`。
- PNG/PDF 重新生成并完成目视检查；Windows `sim_paper_figure` 中的同名 PDF 已同步覆盖，SHA-256 与项目内副本一致。

### 2026-09-02：Breakdown 图高度缩短至 2/3

- 保持宽度、横排布局、stack width 和字号不变，将画布从 `5.20x4.20 in` 改为 `5.20x2.80 in`；PNG 高度由 2548 px 降为 1708 px。新版 PNG/PDF 生成成功并完成目视检查。
- Windows `sim_paper_figure` 目录中的同名 PDF 已覆盖为新版，并通过 SHA-256 与项目内副本核对一致。

### 2026-09-02：Breakdown 图重命名与 PDF 导出

- 绘图脚本默认 basename 改为 `vgg_sana_breakdown_comparison`，项目内成功重新生成 PNG/PDF；旧 basename 产物保留，未做删除。
- PDF 已另存到 `C:\\Users\\ydy\\Downloads\\sim_paper_figure\\vgg_sana_breakdown_comparison.pdf`。项目内与 Windows 副本均为 19 KiB，SHA-256 一致：`544ad32f965a72d60966a52250b7261bb8f4645724920f9c3fa8a7a8116208e6`。

### 2026-09-02：合并 breakdown 图最终横排单栏版

- Hardware/Energy 恢复左右横排；在保持所有文字 `2x` 字号的前提下，最终画布采用 `5.20x4.20 in` 页面单栏宽度。严格 `3.45 in` 横排会导致 15–17 pt 文本不可避免地重叠，因此未保留该不可读版本。
- stack width 增至 `0.90` 并收紧 x 轴留白，SOMA-Sim/SANA-FE 拆为两行标签，图例恢复三列。重新生成 PNG/PDF 并目视确认所有百分比完整、无裁切或互相覆盖。

### 2026-09-02：合并 breakdown 图改为单栏版式

- 图宽由 `7.10 in` 改为论文单栏 `3.45 in`，Hardware/Energy 从左右排列改为上下排列；bar width 从 `0.58` 调整为 `0.50`，图例改为纵向排列。
- 所有文字相对上一版精确放大 `2x`：全局 `16.4 pt`、轴标题 `17 pt`、刻度/图例 `16 pt`、stack 百分比 `15 pt`。重新生成的 PNG 为 `2139x5428`，PDF 同步更新并完成目视检查。

### 2026-09-02：合并 breakdown 图百分比字号统一

- Synapse、Soma、Network 三类色块内的百分比最终统一使用 `7.5 pt` 白色粗体。重新生成 PNG/PDF 并目视确认字号一致、标签可读。

### 2026-09-02：合并 breakdown 图配色微调

- 将 Hardware work / Energy 双栏图中的 Soma 百分比改为白字，所有 stacked segment 及图例色块的边界改为黑线；重新生成 PNG/PDF 并目视确认窄橙色色块数字可读。数据、排序和统计口径均未改变。

### 2026-09-02：Hardware work / Energy normalized 双栏合并图

- 新增 `others/hardware_energy_breakdown_20260902/` 自包含绘图脚本与口径说明；双栏最终图左侧为 serialized hardware work、右侧为 energy，每栏均按 SOMA-Sim 左/SANA-FE 右排列，不显示子图标题。
- 左侧纵轴为 `Hardware share (%)`，右侧为 `Energy share (%)`；图和图例只保留 Synapse、Soma、Network，移除占比小于 `0.01%` 的 Synchronization 和零值 Dendrite。Soma 的窄橙色色块改用居中的 `6.2 pt` 深色标签，避免白色 segment 边界遮挡数字。
- 脚本通过 AST 语法解析并用系统 Python/Matplotlib 3.10.9 成功生成 `4329x1651`、600 dpi PNG 和 PDF；最终图已目视检查。未修改 simulator、原始数据或硬件参数。

### 2026-09-02：Normalized Energy 与 serialized hardware work breakdown

- 将既有 `energy_breakdown_comparison` 精简为单栏 normalized-only stacked bar，保留 Synapse/Soma/Network/Dendrite 分类；重新生成的 PNG 为 `2139x1711`，PDF 同步更新。
- SANA-FE 不直接输出可相加的硬件 latency breakdown，但原生提供 synaptic operations、neuron firing/update、packets、hops 和组件 energy，足以结合架构单次 service latency 重建 serialized hardware work。新图按 Synapse、Soma、Network、Synchronization 分类；该指标是累计忙碌时间，不是并行硬件关键路径 latency。
- 完整 VGG16 serialized work 中，SANA-FE/SOMA-Sim 分别为：Synapse `46.9%/46.9%`、Soma `2.4%/2.3%`、Network `50.7%/50.8%`；Synchronization 均小于 `0.01%`。自包含 CSV、公式说明、脚本及 600 dpi PNG/PDF 位于 `others/serialized_work_breakdown_20260902/`。
- 两个脚本均通过 AST 语法解析并用系统 Python/Matplotlib 3.10.9 成功重绘，两张最终 PNG 已目视检查。未修改 simulator、硬件参数或 benchmark 结果，未生成 `__pycache__`、构建缓存或凭据。

### 2026-09-02：SOMA-Sim / SANA-FE VGG16 Energy breakdown 对比图

- 使用现有完整 VGG16 summary 与 SANA-FE benchmark 数据，统一为 Synapse、Soma、Network、Dendrite 四类；其中 SOMA Network=`axon + router + link`，SANA-FE 使用 `network_energy`，两边 Dendrite 均为 0。
- 在 `others/energy_breakdown_20260902/` 新增自包含 CSV、绘图脚本、README，以及 600 dpi PNG/PDF。左图比较绝对能量，右图比较归一化占比；SANA-FE/SOMA-Sim 总能量分别为 `159.96/163.71 mJ`，SOMA-Sim 高 `2.34%`。
- 脚本经 AST 语法解析并用系统 Python/Matplotlib 3.10.9 成功重绘；PNG 为 `4329x1711`，PDF 可识别，最终图已目视检查总量、百分比、图例和标注无遮挡。未修改 simulator，也未生成构建缓存或凭据。

### 2026-09-01：最终 SANA-style 候选 host latency 重跑

- 为避免 earlier router/5.5 ns ablation 污染，新建干净 `/tmp/soma_sim_host_candidate.yYD5Np` Release 副本，只对当前版本应用归档的 `simulator.hpp/.cpp` 候选补丁，并链接相同 VGG16 输入/权重；正式仓库 simulator 未改。Release 构建和 CTest 1/1 通过。
- 完整 256-step VGG16 成功：177,386,828 packets、prediction/expected=`3/3`。程序内部 `host_latency_s=348.019318405 s`、吞吐 `509,703.969346 packets/s`；`/usr/bin/time -v` 为 elapsed `5:49.48`、user `348.83 s`、system `0.65 s`、CPU `100%`、peak RSS `322,172 KiB`。
- 相对同机当前版本 `264.833935140 s`，proof-of-concept 候选增加 `83.185383265 s` / `31.4104%`，即慢 `1.314104x`。该候选保留原 free-time functional timing 并额外重算 detailed timing，属于双算开销，不代表将来替换旧 NoC timing 后的最终 host 性能。
- 最终候选独立 hardware CSV 合计 `0.595248400859 s`，相对 SANA 总误差 `+5.0852%`、MAPE `6.5734%`、median `4.9784%`、Pearson `0.8466`；原 terminal/summary 的 `0.644343003 s` 仍是未替换的旧统计路径，不能误作候选 hardware timing。
- 最终 256-step CSV 和 summary 已归档为 `others/latency_scheduler_experiment_20260901/data/experimental_exact_source_timestep_metrics_256.csv` 与 `candidate_host_summary.json`；SHA-256 分别为 `4767fc48...` / `8b1ef4d7...`。

### 2026-09-01：VGG16 timestep latency 增长误差与 SANA-style scheduler 实验

- 冻结当前 `simulator/**` SHA 后只在 `/tmp/soma_sim_latency_exp.tLfZ4b` 的 Release 副本做 timing ablation；本轮未 SSH、未用 GPU，也没有把候选 scheduler 应用到当前 simulator。结束时 34 个 simulator 文件逐一通过 SHA 校验，确认工作区热路径保持用户要求的当前版本。
- 解释了 Pool-fused 后误差为何变大：旧显式-Pool 版本只有 SANA 的 `68.018%` synaptic updates / `84.472%` hops，其 workload 缺失与 free-time timing 高估发生抵消；新版本 workload 为 SANA 的 `102.619%` / `102.957%` 后，抵消消失，稳定区 65–256 steps 高约 `11.5%–12.0%`。
- 无 link contention、1 ns pipeline、4.1 ns busy 和 5.5 ns busy 四个控制组证明，仅改逐 link free-time/常数不能同时对齐 warm-up 与稳定段；5.5 ns 虽让前 64 steps 总误差变成 `+2.355%`，MAPE 仍为 `8.308%` 且 timestep 2–5 低 `26.65%`，属于参数拟合而非架构修复。
- 按 SANA-FE detailed scheduler 实现 source-Core FIFO、route density、16-entry path capacity、在途 destination processing mean 和 destination-Core 串行反压后，完整 256-step timing 为 `0.595060785531 s`：总误差 `+5.0521%`、MAPE `6.5692%`、median `5.0515%`、Pearson `0.8472`，分别优于当前的 `+13.7524%`、`14.8177%`、`13.8819%`、`0.7037`。timestep 4 从 `2.245606 ms` 变成 `3.497456 ms`，SANA 为 `3.478378 ms`。
- 前 64 steps 的趋势改善更强：第一版 MAPE `5.2413%`、Pearson `0.9699`；进一步对齐 equal-time comparator 和 source neuron/axon-out 串行后为 MAPE `5.2186%`、Pearson `0.9699`，说明这两个细节不是剩余误差主因。
- 剩余约 5% 与本轮暂缓的功能流量差一致：前 64 steps SOMA updates/hops 比 SANA 多 `2.814%` / `2.969%`，完整运行多 `2.619%` / `2.957%`；逐 step 残差与 update ratio 相关约 `0.51`，额外 firing 经非线性拥塞放大。没有使用 latency scale 掩盖量化/threshold 边界。
- 可复现实验说明、原始 CSV、候选补丁与已目视检查的 PNG/PDF 位于 `others/latency_scheduler_experiment_20260901/`。实验副本构建成功、CTest 1/1 通过；完整功能运行仍处理 177,386,828 packets、prediction=3。绘图脚本通过 `py_compile`，生成物无 `__pycache__`/凭据。

### 2026-09-01：远程更新、VGG16 重跑与 latency figures 重画

- 将当前 Pool-fused/真实 firing time/Conv-Dense latency 分型版本增量同步到远端 `/home/dingyang_yu/comparison/soma_sim`；只覆盖本轮项目文件及正式 VGG 资产，保留远端 `.git`、旧 `output_vgg16_remote_edaa977/` 和用户已有 NOTES，没有运行 GPU workload，也没有修改 `/home/dingyang_yu` 之外的文件。
- 首个 59 MiB SCP 被截断并由 SHA/tar EOF 检出，未在半同步状态构建；随后改用 8 MiB 分片，重组 archive 通过 SHA-256 `348790aa...` 和 `tar -tzf` 后完整覆盖。远端 weights/input SHA 与本地一致，所有传输临时文件已清理。
- 远端 Release build 与 CTest 1/1 通过；新输出 `output_vgg16_remote_pool_fused_20260901/` 完整排空 256 steps。结果为 prediction/label=`3/3`、279 cores/70 tiles、177,386,828 packets、2,210,196,692 hops、5,385,396,144 updates、hardware `0.644343003 s`、host `158.421693818 s`；硬件统计/score 与本地一致，timestep 1 为 `11.7328 us`，17 个 layer rows 中没有 Pool。
- `/home/dingyang_yu/comparison/figure/data/soma_timestep_metrics.csv` 已替换为新远端数据，旧版另存为 `soma_timestep_metrics_edaa977.csv`，SANA-FE CSV 未改。三个原脚本均在 `snn` 环境成功运行，重新生成 6 个 PNG + 6 个单页 PDF；远端 figure 与本地 `others/latency_figure_20260901/` 镜像的关键文件 SHA 一致。
- 新对比指标为总误差 `+13.7523715%`、MAPE `14.8177137%`、median `13.8819222%`、Pearson `0.7036828`，signed error 范围 `[-35.4410121%, +41.2778146%]`。error 脚本改为根据数据动态扩展纵轴，避免旧的 `+30%` 固定上界裁掉峰值；单/双栏六张 PNG 已逐张目视检查，无曲线或标签裁切。
- 远端同版本 host latency 比本地 `264.833935140 s` 少 `106.412241322 s`，当前观测为 `1.671702x`；两边硬件统计一致，但该数字仍会受服务器负载、编译器和 CPU 单核状态影响。

### 2026-09-01：VGG16 Pool fusion、真实 generation time 与 synapse latency 分型

- 保持 Spatial Pattern/source-major 紧凑表示，在资产生成阶段 compose 5 个 AvgPool：`conv2/4/7/10` 改为 6x6/stride2/padding2 融合 kernel，`fc1_weight` 改为 `[2048,512]` 的 Pool→Dense 融合权重；正式 mapping 变为 input + 13 Conv + 3 Dense，无独立 Pool stage，共 279 cores / 70 tiles。
- 调度改为外部 input 在 timestep start 入队，internal firing 的 `generated_time/current_time` 均使用各自 `firing.hw_finish_time`；移除全局 `data_phase_start` clamp，global queue 继续按 `(timestep, generated_time, sequence_id)` 稳定交错处理。
- hardware schema 拆为 `synapse.spatial=3.1 ns` 与 `synapse.dense=3.8 ns`，destination Core 按 layer op 选择。本次完整 VGG 中 spatial/dense updates 分别为 `5,354,148,864` / `31,247,280`，对应 service 总和与 summary 的 `16,716,601,143` cycles（向上取整）一致。
- 验证：fusion helper 数值/形状测试、`python3 -m py_compile tools/prepare_vgg16_assets.py`、`cmake --build build -j2`、CTest 1/1、最小样例及完整 256-step VGG16 均通过。CTest 逐层核对 reference physical-Core 起点/分区，另覆盖 279-Core/无 Pool mapping、两类 latency、Linear/Spatial service time 与 `11.7328 us` 无流量 timestep；最终补强断言后的重建和 CTest 再次通过。
- 完整 VGG16 队列排空：prediction/label=`3/3`，279 cores / 70 tiles，177,386,828 packets、2,210,196,692 hops、5,385,396,144 updates；timestep 1 为 `11,732,800 ps`，hardware latency `0.644343003 s`，host latency `264.833935 s`，score cosine `0.9998356`。layer metrics 只有 input、conv0–12、fc1、fc2、readout，没有 Pool。
- 与 SANA-FE 相比，hardware latency 仍高 `13.7524%`，packets/updates 约高 `2.6%`、hops 高 `2.96%`；这是本轮未处理的量化/threshold 边界与既有 NoC contention 时序残差，未通过调参掩盖。保存的 SANA-FE host call 为 2140.541 s，当前 SOMA 观测快 `8.0826x`，但不是同机测量。
- 生成物检查：正式大权重、输入和 benchmark output 均位于项目约定目录且受 ignore 管理；没有新增构建缓存或凭据。上一轮未跟踪的 `others/` 审计资产保留不删。

### 2026-08-31：SOMA-Sim / SANA-FE timestep latency 论文图

- 在 `/home/dingyang_yu/comparison/figure` 生成逐 timestep hardware latency、signed relative error 和 latency scatter 三类图；每类同时提供 3.45-inch 单栏与 7.10-inch 双栏版本，并输出矢量 PDF 和 600-DPI PNG，共 12 个图文件。
- 保留三个独立可复现脚本：`plot_latency_timeseries.py`、`plot_latency_error.py`、`plot_latency_scatter.py`；每个脚本一次生成本类图的两种栏宽。目录内 README 记录依赖、运行命令、版式和指标定义，两份源 CSV 复制到 `figure/data/`，SHA-256 与原始文件一致。
- 使用 FreeSans/DejaVu Sans、色盲友好蓝/朱红、线型与 marker 双重编码、无顶/右边框和轻量网格；图中使用原始 256-step 数据，不平滑、不删除 warm-up 或异常 timestep。scatter 用橙色菱形明确区分 T=1–8，并保留 `y=x` 参考线。
- 按用户要求将 Matplotlib 3.11.1 安装到 `snn` Conda 环境，后续可直接 `conda run -n snn python <script>` 修改重画。
- 验证：三个脚本均实际运行成功；3/3 AST parse 通过；六份 PNG 均完成目视检查，标签、图例和注释无裁切；六份 PDF 均识别为单页 PDF 1.4。未重跑 simulator benchmark。

### 2026-08-31：SOMA-Sim / SANA-FE VGG16 保真度数据与作图口径分析

- 核对两边现有 summary、逐 timestep CSV、SANA-FE benchmark/NPZ 和统计递增位置；确认逐 timestep hardware latency 可按逻辑 timestep 1–256 直接配对，SOMA 使用 `hardware_end_time - hardware_start_time`，SANA-FE 使用单步 `chip.sim(1)` 的 `sim_time`。
- 当前 256-step 数据总 latency 为 SOMA `0.5678836707 s`、SANA-FE `0.5664435778 s`，总量相对差 `+0.2542%`；但逐步 Pearson `0.6587`、MAPE `6.0796%`、中位绝对百分比误差 `4.5278%`，最大相对误差在 timestep 4（`-51.3208%`）。说明总量误差存在正负抵消，逐步曲线应与 signed relative-error 子图一起展示，不能只用总和宣称逐步精确。
- 可直接复用的更强功能证据包括最终 10-class scores 和 SANA-FE 保存的 `output_potential[T,10]`、各 Conv/FC 的逐 neuron/逐 timestep firing tensor；SOMA 当前只导出最终 scores 与聚合 layer/timestep metrics，后续若需强验证，应增加逐 timestep output potential、`layer x timestep` firing count，以及可选稀疏 firing trace/hash。
- 统计口径限制：SANA-FE `spikes` 是展开连接后的 synapse-input 次数，`neurons_fired` 才是 firing，`packets_sent` 是 packet；SOMA `processed_spikes` 当前等于 packet，不能直接与 SANA-FE `spikes` 比。当前 SOMA 保留显式 AvgPool、SANA-FE 融合 pooling，因此 firing/packet/update/hop 和能耗 breakdown 在图结构统一或明确归一化前只适合诊断，不宜作为“等价”主证据。
- 验证方式：只读检查本项目与 `/home/dingyang_yu/comparison/sanafe_vgg16_snn_cifar10`、`SANA-FE/src` 中现有结果和统计定义，并用标准库脚本计算误差；未重跑 benchmark、未使用 GPU、未修改项目外文件。

### 2026-09-01：VGG16 逐 timestep hardware-latency 残余误差审计

- 对照当前 `edaa977`、SANA-FE detailed scheduler 和远程 256-step CSV/图完成只读审计；未修改仿真代码、未修改远程文件、未运行 GPU。详细分析副本在 `others/latency_figure_20260901/`、`others/sanafe_src_20260901/` 和 `others/sanafe_vgg_audit_20260901/`。
- 原 6 项中，按 local updates 计 synapse、physical-Core packetization、per-Core soma 并行/max、Tile/XY multi-hop 和 timestep sync 等基础机制都已进入 runtime；其中 soma 串行粒度与 sync 可判定已修复，其他项仍受下述配置/调度差异影响。
- 图/映射仍未对齐：SOMA 的 5 个显式 Pool 额外引入 31,232 neurons/31 cores/8 tiles，从 global core 134 起就与 SANA-FE 融合图错位。总量 SOMA/SANA 比为 updates `68.02%`、packets `84.72%`、hops `84.47%`、firing `111.32%`。
- 最主要的 timing 残差是 source scheduling/NoC：SOMA 将全部 firing 压到全局 `data_phase_start` 后发送，丢失 per-Core neuron generation spacing，并使用逐 hop free-time；SANA-FE 保留 per-Core `generation_delay` 并用 detailed route-density heuristic。这会改变 burst/overlap/contention，不是等价的实现差异。
- 次要残差是参数与 neuron arithmetic：SOMA 统一用 3.1 ns synapse，而 SANA-FE Dense 用 3.8 ns；SANA-FE 还使用 1/64 膜电位量化和严格 `>` threshold，SOMA 当前没有量化且使用 `>=`。
- 数据结论：总 latency 误差 `+0.254234%` 是正负抵消，不代表逐 step 精度；MAPE `6.0796%`、median `4.5278%`。timestep 2–5 低估 `36.98%`，6–16 高估 `13.36%`，17–32 高估 `13.82%`。timestep 1 两边均为 `11.7328 us`，证明 soma/sync 基线正确。
- 本轮验证为 `cmp`/字段级 CSV 一致性、Python 只读汇总/相关分析、代码路径审计与远程 YAML 只读复制；本轮无源码变更，因此未重复构建或仿真。未覆盖边界是：没有生成“同图+同调度”的控制组重跑，所以 neuron 量化、Dense latency 和 NoC 三者的独立 critical-path 贡献尚未分离。

### 2026-08-31：VGG16 physical Core packet timing

- 新增 Tile/physical Core 地址层；mapping 按 `max_neurons=1024` 将 layer 连续拆分，并用 `physical_neuron_order: channel_major` 对齐 SANA-FE 空间层的 Core 切分。当前显式-pool SOMA 图映射为 310 cores / 78 tiles。
- 一个 firing 按其连接触达的 destination Core 集合生成 packet；packet route 由 source/destination Tile 生成确定性多跳 XY 路径，继续使用紧凑 router output/link free-time 表，不实例化 Router 或 synapse 边。
- destination Core 使用 Spatial Pattern 只遍历本 Core local updates，并按 `N_local_updates * synapse_latency` 串行占用 Core；soma loop 每个 physical Core 最多 1024 neurons、Core 间并行。
- 解析并应用 timestep synchronization latency table；70/78 mapped tiles 均取 1.8 us。无流量满 Core timestep 精确为 `1024 * (6.0 ns + 3.7 ns) + 1.8 us = 11.7328 us`。
- summary/CSV 新增 physical core/tile、packets、NoC hops、synaptic updates、逐 timestep hardware latency，以及 soma/synapse/NoC/synchronization breakdown。
- 验证：Release build 与 CTest 1/1 通过；最小样例完成。完整 VGG16 处理 146,488,599 packets、1,813,376,089 hops、3,569,567,135 updates，hardware latency `0.5678836707 s`，相对 SANA-FE `0.5664435778 s` 为 `+0.2542%`；prediction/label 均为 3，host latency 206.031 s，peak RSS 约 203 MiB。对比已保存的远程 SANA-FE host time 2140.541 s，当前 SOMA 观测快 `10.3894x`，但两数据非同机且 profiling 设置不同。

### 2026-08-31：Loihi / SANA-FE timestep-synchronous 两阶段执行

- 同步模式改为“读取上一 timestep buffer 的 neuron processing → 当前 Data/NoC/synaptic accumulation → queue drain/barrier”；Data path 不再访问 soma state，Core buffer 用独立 pending 位保留累加和为 0 的输入。
- 删除 `SpikeKind` 与 Bias queue event；global queue 只保存真实 Data。bias/leak/membrane transition 在按 neuron id 升序的 loop 中处理，每个 neuron 每 timestep 最多 firing 一次，soft-reset 剩余电位留到后续 timestep。
- hardware timing 将原 `soma_update: 9.7 ns` 拆为 `soma_access: 6.0 ns` 和 `soma_update: 3.7 ns`。每个 mapped neuron 收取 access，只有实际 state update 再收取 update，firing 继续单独收取 `soma_fire`。
- 所有 Core 从同一 timestep 起点并行执行 neuron phase；Data phase 从最慢 Core 完成后开始。固定运行到 input CSV 的最后 timestep，不自动增加 pipeline flush step。
- 验证：Release 构建和 CTest 1/1 通过，覆盖 Data-only accumulation、下一步 processing、mapped/updated timing、内联 bias、每步单 firing 和剩余膜电位延后 firing。最小 2-step 样例处理 5 个 Data spike、hardware latency `261800 ps`、预测 0。
- VGG16 256-step 全量运行完成：处理/发射 5,176,477 个 Data spike，hardware latency 0.2253739237 s，host latency 10.4269 s，peak RSS 约 132 MiB；预测/标签/参考均为 3，score cosine 为 0.9996414。

### 2026-08-31：敏感背景文件 Git 历史清理

- 将 `docs/background.md` 加入 `.gitignore` 并停止 Git 跟踪，本地文件保持不变；安全修复提交未混入现有其他未提交代码。
- 确认该路径曾存在于已推送历史后，在隔离临时克隆中重写 `main` 全部历史并删除该路径；清理后所有可达 ref 均不再包含该文件或凭据标记。
- 使用带旧远程 SHA 租约的 `--force-with-lease` 推送成功，GitHub `main` 与本地 `main` 均更新到清理后提交 `dc6537b`；远程没有其他 branch/tag 需要处理。
- 验证：本地文件存在且被 `.gitignore` 命中，`git ls-files docs/background.md` 为空，`main == origin/main`，当前其他未提交改动完整保留。本轮仅修改 Git 元数据和 ignore 规则，未运行构建测试。
- 安全边界：远程历史清理不能撤销他人已经克隆、缓存或 fork 的内容，相关服务器密码仍必须轮换。

### 2026-08-31：HardwareResource 抽象重命名

- 将通用 free-time 资源抽象从 `buffer.hpp/.cpp` / `BufferResource` 重命名为 `hardware_resource.hpp/.cpp` / `HardwareResource`，同步更新 CMake、Core、Memory、Simulator 和 PROMPT 引用。
- 保留 `ResourceReservation`、已有 free-time 注释和所有 reservation 时序行为；Release 构建、CTest 1/1 与最小样例通过，结果仍为 6 个 data spike、`412900 ps`、预测 0。

### 2026-08-31：删除 SomaDrain event

- 删除 `SpikeKind::SomaDrain`、Core drain API/状态和 Simulator 二次调度；Data/Bias 对 Core 的同一次更新现在直接完成 threshold check、reset 与 firing。
- 候选集只包含本次实际越阈值的 neuron，不做全状态扫描；firing 按 neuron id 升序逐次占用配置中的 `soma_fire` resource/latency，并在各自完成时刻作为普通 Data spike 入全局队列。
- 保持 global queue、NoC/Core resource-free-time、energy 和 processed/emitted statistics 口径；state update 仍记录在 update 完成时刻，output spike 仍记录在 firing 完成时刻。
- 验证：Release 构建和 CTest 1/1 通过，新增有序 soft-reset firing 及逐次 `soma_fire` 时延断言；最小样例保持 6 个 data spike、`412900 ps`、预测 0。
- VGG16 256-step 全量队列排空：处理/发射 5,336,573 个 spike，hardware latency 8.8170754285 s，host latency 21.4050 s，peak RSS 约 132 MiB；预测/标签/参考均为 3，score cosine 为 0.9988768。

### 2026-08-31：timestep synchronization 注入与时间推进

- 在 `hardware.yaml` / `HardwareConfig` 中增加 `execution_mode: timestep_synchronization`，当前 Loihi-style 配置显式启用；当前版本拒绝未实现的其他 mode，但没有加入占位分支。
- input spike 的逻辑 timestep 改为从 1 开始，生成器和现有 CSV 的 `generated_time/current_time` 均为 0；删除 1 秒的人工 timestep period。
- simulator 只注入当前 timestep，完整排空其 Data/Bias、NoC 和内部派生事件后，再从实际硬件完成时刻注入下一步；派生 spike 始终继承原逻辑 timestep。global queue、Core/NoC resource-free-time 和模块边界保持不变。
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

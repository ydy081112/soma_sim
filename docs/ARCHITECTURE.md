# MVP architecture

一次 data spike 的路径固定为：virtual/physical source core 的 axon-out、mapping 静态 route、destination input buffer、synapse SRAM、Spatial Pattern Template 累加、SoA soma state。Router 不是对象；`router_id * 5 + output_port` 直接索引 output/link 两张 `SimTime` 数组。

队列只按 `(generated_time, sequence_id)` 排序。`generated_time/current_time` 是目标硬件时间轴上的 timestamp；`current_time` 在该 spike 穿过资源时单调增加，新生成的 spike 以完成时刻重新入队。硬件服务/传播/拥塞 duration 统一称为 `hardware_latency`，所有配置值在启动时变为 ps，因此热路径没有单位或浮点时间换算。

主机执行事件循环所花的 wall-clock duration 统一称为 `host_latency`，不得与 `hardware_latency` 相加或互相替代。对外 JSON/CSV 使用完整名称；C++ 热路径使用 `hw_*` 简写。

每次 synapse 更新只把首次越过 threshold 的 neuron id 放进候选队列。内部 `SomaDrain` fake spike 每次最多发射一个 neuron；若还有候选或 soft-reset 后仍超过 threshold，就把新的 fake spike 放回全局队列。这避免 time-driven 的逐 tick 全状态扫描，并保持主循环一次处理一个事件。

带 bias 的层每个输入逻辑 timestep 只有一个 `Bias` 事件，它连续更新该层 SoA state 并复用相同候选队列；因此 bias 不会退化为逐物理 tick 扫描。

当前 MVP 的 layer mapping 可以表达一个 layer/partition 对应一个逻辑 Core。大层可以先以 `partition: aggregated` 做架构级快速估计；精确的跨 Core partition fan-out 是后续编译栈应生成的多目标 mapping，不在本版自动推断。

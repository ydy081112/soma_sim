# MVP architecture

一次 Data spike 的路径固定为：virtual/physical source core 的 axon-out、mapping 静态 route、destination input buffer、synapse SRAM、Spatial Pattern Template 累加到 Core 的 `timestep_buffer`。Data path 不访问 SoA soma state。Router 不是对象；`router_id * 5 + output_port` 直接索引 output/link 两张 `SimTime` 数组。

全局队列的排序语义由 `hardware.yaml` 的 `architecture.execution_mode` 控制。当前 Loihi-style 配置启用 `timestep_synchronization`，队列只保存真实 Data spike，并按 `(timestep, generated_time, sequence_id)` 排序。一个 timestep 的 Data、NoC 和 synaptic accumulation 全部排空后才越过 barrier；配置边界仍允许后续增加无 barrier 的异步执行语义。

同步模式的 input CSV 只保存从 1 开始的逻辑 timestep，`generated_time/current_time` 统一为 0，不再用预写的绝对时间隔离 timestep。运行时 `generated_time/current_time` 才表示目标硬件时间轴上的 timestamp；`current_time` 在 spike 穿过资源时单调增加，新生成的 spike 以完成时刻重新入队。硬件服务/传播/拥塞 duration 统一称为 `hardware_latency`，所有配置值在启动时变为 ps，因此热路径没有单位或浮点时间换算。现阶段只实现该执行模式，但排序器和硬件配置的边界保留了后续无 barrier 架构采用其他时间语义的空间。

主机执行事件循环所花的 wall-clock duration 统一称为 `host_latency`，不得与 `hardware_latency` 相加或互相替代。对外 JSON/CSV 使用完整名称；C++ 热路径使用 `hw_*` 简写。

同步模式中，timestep `t` 先让所有 Core 从同一硬件时刻开始 neuron processing：按 neuron id 升序读取 `t-1` 留下的 buffer，访问全部 mapped neurons，只对 pending input、非零 bias 或已有 membrane state 的 neuron 做实际 state update。每个 neuron 每步最多 firing 一次，soft reset 后剩余电位留给后续 timestep；产生的 spike 标记为 `t`。最慢 Core 完成后进入 Data phase，内部 firing 与外部输入一起经过 axon-out/NoC/synapse，并只写入 `timestep_buffer`，供 `t+1` 使用。

Bias 是 neuron-processing loop 中按 channel 读取的固定参数，不经过 axon、NoC 或 synapse，也不创建 queue event。Core neuron phase 的服务时间为 `N_mapped × soma_access + N_updated × soma_update + N_fired × soma_fire`；当前配置把原 9.7 ns 拆成 6.0 ns access 与 3.7 ns update。

当前 MVP 的 layer mapping 可以表达一个 layer/partition 对应一个逻辑 Core。大层可以先以 `partition: aggregated` 做架构级快速估计；精确的跨 Core partition fan-out 是后续编译栈应生成的多目标 mapping，不在本版自动推断。

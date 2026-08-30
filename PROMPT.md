在开始前你需要知道soma_sim这个项目的背景，这个是一个snn simulator的项目，具体的背景我放在了docs/background.md，你先仔细阅读。
你首先需要做的是，构建AGENTS的flow，在AGENTS.md里面写相应的东西，然后docs里面加一些NOTES.md等等这些，记得多记构建的阶段的NOTES，因为网络任务有可能随时会中断，NOTES.md里面要分成[进行中]和[已完成]。然后在没有必要的时候不要做smoke test，少做各种测试。
警告：我开了yolo模式你在删除/修改任何本项目之外的东西一定要小心谨慎，没有必要的话不要修改本项目外的东西！实在要动你可以复制进来放个others文件夹来修改！
最高级警告：1. 要是需要用到里面的GPU需要先 nvidia-smi 看看有没有空闲的GPU，不能停掉别人的进程； 2. 不得删除，移动，更改任何出了/home/dingyang_yu外面的文件！
下面，请帮我先从零实现一个纯 C++ 的可配置 SNN architecture simulator MVP（Minimum Viable Product），先不要 Python/pybind11，C++17 + CMake等等，以及先不要实现任何mlir编译栈的东西，代码必须分文件、中文注释清晰。
# 模拟器的输入包括：
1. /arch/hardware.yaml：硬件配置，第一版按 Loihi/SANA-FE 风格建模 PE/Core、NoC、router hardware latency、link hardware latency、buffer、synapse/SRAM、soma/neuron 参数，但所有参数必须可配置，不能把 Loihi 写死。你参照/home/ydy/compiler/SNN-HW-Sim/input-hardware/ours_chip/hardware.yaml这个风格来做，但是里面的数据你完全参照/home/dingyang_yu/comparison/SANA-FE/arch/loihi_large_vgg.yaml里面的数据来填，目前这一版的风格，假如前续版本里面的yaml不够好，你可以参照sana-fe去做
2. /compiler/mapping_output/mapping.yaml：layer/partition 到 PE/Core 的 mapping，以及静态 NoC route。这个参照前续版本的/home/ydy/compiler/SNN-HW-Sim/compiler/layer-to-pe/mapping_mlp4.yaml这个文件去构建，目前还没有编译栈先根据要求你来构建
3. /input/weights.npz：使用 Spatial Pattern Template + source-major weight，不展开 neuron-to-neuron connections。Conv 权重布局 [Cin,Kh,Kw,Cout]，运行时使用 plan_pattern_id / plan_dst_base / pattern_ptr / pattern_dst_offset / pattern_weight_offset。代码已经写好，放在/home/ydy/compiler/soma_sim/compiler/connectivity/vgg_spatial_template_compiler.py，你可以根据项目具体细节更改
4. /input/input_spike.csv：由数据集编码生成的输入 spike，这个版本你先用cifar10的一张图片把他转换成这个.csv。需要包含：
generated_time, current_time, spike_id, layer_id, src_neuron, src_pe, src_router, dst_pe, dst_router, value等等。具体你可以根据模拟器的代码逻辑来。src_router/dst_router/route 原则上由 mapping.yaml 和 routing runtime 决定，CSV 中即使出现也仅作为 debug/trace 字段，不能成为 routing 的唯一真值来源。第一版 CIFAR10 input encoder 使用可配置 rate coding。
# 核心仿真机制：
- 先实现单线程，一次循环只处理一个 spike。
- 输入 spike 采用 SANA-FE 风格的 virtual input/source core 抽象：外部输入 spike 由映射在 virtual input PE 上的 source neuron 产生，然后通过 axon-out 进入 NoC
- 一个全局 SpikeQueue，按 (generated_time, sequence_id) 排序。
- 每次取最早的 spike，从 source PE 沿 mapping.yaml 中的静态 route 传输到 destination PE。
- 不显式创建大量 Router 对象。
- NoC 用紧凑的资源表，例如：
router_output_free_time[router_id][output_port]。当然这个资源表需要做成根据架构的不同，router的不一样的，做成不一样的。目前就按照sana的来，他有一个switch然后1个VC
- spike 每经过一个 hop/link：
start=max(arrival_time, link_available_time)；
wait=start-arrival_time 计入 router congestion hardware latency；
再累加 router/link hardware latency，并更新该 output port 的 free time。
- 配置硬件细节的时候，不要说某个函数是loihi的这样子，比如模拟器这次的Link是异步握手的，你就在yaml里面的特性就可以写成:
  send: 
    req: true
    data:true
  Receive:
    ack: true
  这样子
  就是看有没有某个信号线，来确定是同步还是异步的，反正就是尽量不要通过loihi这种芯片的名字来说用哪个micro-arch这样子
- 这个Link你可以参照前续版本里面的router.py里面的写法，分成input_to_output等等这些阶段
- 到达 destination PE 后执行：
input buffer → synapse/weight SRAM → neuron state update → threshold。
     先不用像前续版本里面一样加一些gated的机制
- connectivity 不展开。SynapseEngine 根据 source neuron 从 Spatial Pattern Template 直接得到 destination spatial block 和连续 Cout 权重块。这里尽量写的省时间一点，因为几十万数量级的spike，每个都是需要取存的，仿真时间会很多的
- neuron state（threshold/voltage这些） 使用连续数组/SoA，不创建 per-neuron C++ object。记住，这个项目的主要目标是快速仿真，设计的时候代码写法以这个为目标来做
- 对于neuron state不要每次是spike都检查一次，只有spike进到core里面加上去，才检查，所以为了放置最后一个spike激活了但还有剩余的voltage大于电压（因为一个spike只会触发一个新的spike，不能触发多个），你就设置一个fake spike在队尾。这个fake spike我在这里不多讲了，你需要到前续版本里面自己去看，一定要仔细看然后实现这个fake spike。他的主要作用就是不要让每次都去检查一遍这个state有没有超过阈值，这样就变成time-driven simulator了
- 若 neuron 达阈值产生新 spike，根据 mapping/static route 构造新的 Spike 并加入全局 SpikeQueue。
- 模拟器内部统一使用整数 SimTime，例如 1 tick = 1 ps。hardware.yaml 中所有 ns/s 等时间在加载时一次性转换成 SimTime。
  例如：
  using SimTime = uint64_t;
  然后：
  6.5 ns → 6500 ps
# 文件路径建议拆成（当然可以不仅限于下面这些）：
simulator/src/
config/
- hardware_config.hpp/cpp
- mapping_config.hpp/cpp
runtime/
- weight_store.hpp/cpp
- spatial_template.hpp/cpp
sim/
- spike.hpp
- spike_queue.hpp/cpp
- simulator.hpp/cpp
- stats.hpp/cpp
hw/
- noc/router.hpp/cpp  放和前续版本router.py类似的东西
- noc/route.hpp/cpp 放和前续版本mesh.py类似的关于路由算法的东西
- core.hpp/cpp 类似前续版本的pe.py
- synapse.hpp/cpp   pe.py里面的小部件的函数
- soma.hpp/cpp   pe.py里面的小部件的函数
- memory.hpp/cpp
- buffer.hpp/cpp
main.cpp
input_encoder.hpp/.cpp
# Statistics 至少统计（专门放一个文件夹/output，可以参考前续版本）：
hardware_latency
- target hardware simulation latency
- total spikes / 逐层的spikes
host_latency
- host_latency_s
- host_processed_spikes_per_sec
- 逐层的host_processed_spikes_per_sec
- 逐timestep的host_processed_spikes_per_sec-这个可以做成csv，类似于/home/dingyang_yu/comparison/sanafe_vgg16_snn_cifar10/sanafe_vgg16_timestep_metrics.csv
hardware_latency 的五个 breakdown（cycles）
- pe_inject_cycles
- pe_compute_cycles
- noc_traversal_cycles
- router_congestion_cycles
- Link busy
energy
- 各组件 energy
- total energy
- output_scores和expected_output

第一步先测试服务器能不能正常连接，然后直接创建完整工程骨架的第一版代码，并附一个最小 hardware.yaml、mapping.yaml、input_spike.csv 示例和测试。不要过度设计，优先保证数据结构简单、高效、后续可扩展。
这一版的目标，你需要仿真一个vgg16 cifar10在loihi上的结果出来，记得复用远程服务器里面的模型/数据集，尽量不要自己花时间训练下载什么的。

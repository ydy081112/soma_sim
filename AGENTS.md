# SOMA-Sim 协作流程

## 项目边界

- 本项目是纯 C++17、事件驱动、可配置的 SNN architecture simulator。
- 当前阶段不引入 Python binding、MLIR 或多线程事件调度。
- 只修改 `/home/ydy/compiler/soma_sim` 内的文件。外部仓库和远程服务器仅允许只读检查；需要复用的资产复制到本项目后再处理。
- 使用远程/GPU 前先查看 `docs/background.md`；运行任何 GPU 工作前必须先执行 `nvidia-smi`，不得停止或影响其他用户的进程。

## 每次工作的 Flow

1. 先完整阅读 `PROMPT.md`、`docs/background.md` 和 `docs/NOTES.md`。
2. 用 `git status --short` 与 `rg --files` 确认现状，保留用户已有改动。
3. 在 `docs/NOTES.md` 的“进行中”记录本轮目标、假设和下一步。
4. 按 `配置 -> 运行时数据 -> 硬件时序 -> 仿真调度 -> 输出统计 -> 工具/样例` 的依赖顺序实现。
5. 修改期间只运行与当前改动直接相关的构建或测试；除非存在跨模块风险，不做重复 smoke test。
6. 阶段完成后，把对应记录从“进行中”移到“已完成”，写明验证命令、结果和未覆盖边界。
7. 交付前检查生成文件没有混入构建缓存、大型临时文件或凭据。

## 实现约束

- 仿真时间统一使用 `SimTime = std::uint64_t`，单位为 ps；配置中的时间只在加载时转换一次。
- 全局队列按 `(generated_time, sequence_id)` 稳定排序；主循环一次只消费一个事件。
- 输入由 virtual input PE/source neuron 注入，route 的唯一真值来自 `mapping.yaml`，CSV route 字段仅用于 trace。
- NoC 使用紧凑资源表维护 router output/link 可用时刻，不创建 per-router 事件对象。
- 神经元状态采用 SoA 连续数组，不创建 per-neuron C++ 对象。
- connectivity 使用 Spatial Pattern Template + source-major 权重，不展开 neuron-to-neuron 边。
- fake spike 是队列中的内部事件，用于延迟扫描/排空 soma fire 请求；不能退化为每个 tick 全量扫描。
- 配置描述能力/信号线，不用具体芯片名在代码中选择微架构。
- 源码按模块分文件，公共接口与关键时序逻辑使用简洁中文注释。

## 验证原则

- 单元测试优先覆盖：时间单位转换、队列稳定顺序、静态 route、NoC 争用、模板寻址、fake spike 排空。
- 端到端测试使用仓库内的小样例；完整 VGG16 运行是单独的 benchmark，不作为每次构建的默认测试。
- 构建目录固定使用项目内 `build/`，输出固定写入 `output/`。

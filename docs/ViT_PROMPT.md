在当前 /home/ydy/compiler/soma_sim 中直接完成 ViT 的 Loihi-like/SOMA 仿真支持并实际跑通一次。目标硬件按 Loihi2-like 形态建模，即 Core 具有可编程 compartment state（Cx State），可保存并读取 neuron 的 persistent state。

NIR 源文件位于：

/home/ydy/compiler/mlir-attention/nir/full_blocks_until_silent_cifar10

下面有 6 个 block，目前只处理 block00，其他 block 不要动。

要求增量式修改现有 SOMA：
- 不做 MLIR；
- 不大规模重构；
- 不写死具体 shape / node 名 / timestep；
- 模型路径、block、输入 tensor、mapping 策略等做成配置项；
- 保留现有 VGG/ResNet 流程。
  
修改完成后重跑 VGG，要求 hardware latency/energy/packet/NoC 等结果不发生变化；host latency 只允许因为新增少量配置判断而增加几秒。

当前已知 block00 的 Transformer 输入 x 已经保存在 NIR 中，而且本身就是 spike：

shape = [31, 1, 65, 128]
语义 = [T, B, token, embedding]

位置：

/home/ydy/compiler/mlir-attention/nir/full_blocks_until_silent_cifar10/block_00_until_silent_t31_integer_lcc.nir

q_IF/metadata/input

它已经是 Patch Embedding + 前端之后的 spike，因此：
- 不做 Patch Embedding；
- 不下载其他 pretrained model；
- 不再做 rate / Poisson / ST-BIF encoding；
- 直接从 NIR 的 x 生成 SOMA input_spike.csv。
  
新增配置驱动的 ST-BIF neuron model，关闭 LIF leak，并严格按照 NIR 实际参数维护 membrane、tracer、threshold/reset 和 signed firing，不要只修改 threshold 判断。

输入 flatten：

src_neuron = token * embedding_dim + channel
timestep = NIR timestep + 1

当前逻辑上有 65*128=8320 个 virtual input neurons，但代码禁止写死 8320/65/128/31，全部从 NIR tensor shape 自动推导。

input layer 只建立一份：

                q_IF
               /
input(x) ------ k_IF
               \
                v_IF

不要复制三份输入。继续沿用现有 virtual/direct input 注入语义，不要为了 ViT 修改 VGG 当前的 input injection 行为。

同时完成 block00 的 NIR → SOMA runtime 数据转换：

- 从 NIR 读取 block00 实际 node、edge、weight、bias、threshold、multiplier、shifter、tensor shape 等 metadata；
- 生成 SOMA 使用的 .npz；
- 普通 q/k/v/proj/fc1/fc2 等静态 projection 继续使用现有 Spatial Pattern / source-major weight 思路，不展开 neuron-to-neuron connectivity；
- residual/norm 必须根据 NIR 的真实 graph 和参数实现，不要仅凭 node 名猜语义；
- qk_multi_IF / qkv_multi_IF 如果现有 Dense/Spatial 无法表达，则只做它们所必需的最小 runtime 扩展；
- 保持现有 global SpikeQueue、Core、NoC、timestep synchronization 框架不变；
- 每个 physical Core 仍受 max_neurons=1024 限制，不够就继续增加 Core。
  
新增配置文件，例如：
configs/vit_loihi_like.yaml


对于 qk/qkv（最重要的算子）：

不要静态展开完整 attention connectivity。根据 block00 NIR 的真实 tensor layout 自动推导 head / token / dim / reduction 维度，禁止写死 4/65/32。

硬件按 Loihi2-like 建模。ST-BIF neuron 的 membrane 和 tracer 都放在本 Core 的 local Cx State 中；tracer 是每个 neuron 一个 persistent scalar state。ST-BIF 不使用 LIF leak，正常产生 signed spike，并通过现有 SpikeQueue/NoC 发送。

QK/QKV 不允许跨 Core 直接读取其他 Core 的 tracer。

QK region 自己维护：

shadow_Q
shadow_K

QKV region 自己维护：

shadow_Attn
shadow_V

这些 shadow state 都是 local persistent state，只通过收到的当前 logical timestep spike 增量更新；不要在 NoC 上传输 dense tracer tensor，也不要保存完整历史 spike。

QK 对齐同一 logical timestep 的 Q_t / K_t，使用 old shadow state 计算：

ΔA_t =
    S_Q,t-1 @ K_t^T
+Q_t @ S_K,t-1^T
+Q_t @ K_t^T
  
计算完成后：

shadow_Q += Q_t
shadow_K += K_t

再将 ΔA_t 累加到 qk_multi_IF，并按照 qk_multi_IF 的 NIR multiplier/shifter/threshold 等实际整数语义执行 ST-BIF。

QKV 同理，对齐同一 logical timestep 的 A_t / V_t：

ΔO_t =
    S_A,t-1 @ V_t
+A_t @ S_V,t-1
+A_t @ V_t
  
然后：

shadow_Attn += A_t
shadow_V    += V_t

再送入 qkv_multi_IF 的 ST-BIF neuron。

QK/QKV 的两个 operand 如果物理 pipeline depth 不同，必须通过 mapping connection delay / timestep buffer 按 logical timestep 对齐，不能按照 packet 的物理到达顺序错误配对。

当前 timestep 的 Q/K/A/V 只保留 sparse transient events，operator timestep 完成后清空；membrane、ST-BIF tracer、shadow tracer 跨 timestep 保留，sample/inference 结束时统一 reset。

如果 qk/qkv 超过 max_neurons=1024，则正常拆成多个 physical Core。每个 partition 只维护自己计算所需要的 local shadow state；如果同一个 Q/K/A/V spike 需要 fan-out 给多个 partition，则通过现有 NoC 正常发送，并真实计入 packet、hop、congestion 和 energy。

不要在 simulator.cpp 中按 q_IF / k_IF / qk_multi_IF 等具体名字写死逻辑，做成配置驱动的通用 operator，例如：

operator_type: incremental_spike_matmul

Loihi-like timing/energy 继续保持现有 SANA-FE 粒度，不单独模拟 tracer/Cx State SRAM read/write：

sram_read  = 0 ps
sram_write = 0 ps

attention.qk_process  = 3.8 ns / attention update
attention.qkv_process = 3.8 ns / attention update

attention.qk_process_energy  = 35.5 pJ / attention update
attention.qkv_process_energy = 35.5 pJ / attention update

其中 1 个 attention update 定义为：

对 1 个 destination accumulator 的 1 次 scalar accumulation。

计时方式与现有 synapse 的：

updates * latency_process

保持一致。

shadow-state access、incremental arithmetic 和 accumulation 全部折叠进 3.8 ns / 35.5 pJ 的 attention process cost，不额外增加 tracer SRAM read/write latency/energy。

普通 ST-BIF neuron 继续使用：

soma_access = 6.0 ns
soma_update = 3.7 ns
soma_fire   = 30.0 ns

不要因为 tracer 再额外增加一次 soma_access。

新增最小 correctness test：使用小规模包含 +1/0/-1 spike 的 tensor，验证 incremental QK：

sum_{τ<=t} ΔA_τ
==
S_Q,t @ S_K,t^T

QKV 同理验证：

sum_{τ<=t} ΔO_τ
==
S_A,t @ S_V,t

必须逐 timestep 一致。


mapping 目标：

- Q/K/V 尽量邻近；
- QK/QKV 紧跟 attention region；
- proj/residual/MLP 后续连续放置；
- 遵守 arch/hardware.yaml 中 max_neurons=1024 等资源约束；
- 尽量复用现有 Loihi hardware.yaml 参数；
- route 继续使用现有确定性 XY routing。
  
  
生成例如：

compiler/mapping_output/vit_block00_mapping.yaml

input/vit_block00_input_spike.csv

以及需要的 ViT runtime .npz / config 文件。

统计方面保留：

summary.json
timestep_metrics.csv
layer_metrics.csv

completed=true 不是正确性的充分条件。

如果 NIR 中存在 node metadata/output，则至少逐 timestep 对比：

q_IF
k_IF
v_IF
qk_multi_IF
qkv_multi_IF

的输出 spike，确认 signed spike 与 NIR reference 一致；若由于 NIR 格式限制无法全部直接对比，至少完成 qk/qkv incremental correctness test，并在最终报告里明确说明实际验证到哪一层。


完成代码后不要只生成文件，直接实际运行一次 block00 的 soma-sim。


如果运行失败，继续定位和修改直到 block00 完成，不要停在“代码已经写好”。


最后写一个 .md 详细汇报，只汇报：

qk_multi_IF 和 qkv_multi_IF 最终在 SOMA 中怎么实现；
ST-BIF membrane/tracer 和 QK/QKV shadow state 分别放在哪里；
block00 使用多少 physical cores / tiles；
增量式、可配置地修改了哪些文件；
correctness 验证结果；
summary.json 中 latency、energy、packets、NoC hops、synaptic updates、attention updates；
各 layer 最主要的开销；
VGG 修改前后 regression 对比，确认 hardware metrics 未变化
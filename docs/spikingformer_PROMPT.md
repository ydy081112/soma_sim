完成 **ImageNet pretrained Spikingformer 的 SOMA 仿真并实际跑通一个 ImageNet sample**（一张图片即可）。

要求可配置地增量修改现有实现，优先复用已有 ViT 的 global SpikeQueue、NoC、mapping、timestep synchronization、timestep buffer 和 attention infrastructure；**不经过 MLIR/编译器，不大规模重构，不破坏 VGG/ResNet/ViT。**

### 1. PyTorch 预处理与 Golden Trace

使用官方 `TheBrainLab/Spikingformer` ImageNet model/config/pretrained checkpoint，T=4。

官方模型 LIF 写死 `backend='cupy'`，CPU golden trace 时改为等价的 `backend='torch'`，**只切换 backend，不改变模型数学语义**。

仿真边界严格放在 Tokenizer 第一个 LIF 输出之后：

```text
ImageNet image
→ proj_conv
→ proj_bn
→ proj1_lif
================ SOMA boundary ================
→ maxpool1
→ proj1_conv+bn
→ proj2_lif
→ ...
```

Python 中：

1. 官方 ImageNet preprocessing 读取一张 224×224 图片；
2. 按官方 forward 将同一图片重复 T=4；
3. 执行 `proj_conv → proj_bn → proj1_lif`；
4. 提取 `[T,C,H,W]` binary spike；
5. 按现有 virtual/direct input 机制生成 SOMA `input_spike.csv`，timestep 从 1 开始，直接作为 global SpikeQueue 初始 spike，**不要 rate/Poisson encoding**；
6. 完整运行 PyTorch，保存后续关键节点逐 timestep golden trace；
7. 导出 Conv/Linear 权重，推理 BN fold 到前驱 Conv/Linear。

路径全部 CLI 配置，不写死机器路径。

### 2. Tokenizer

SOMA 从 `maxpool1` 开始。

当前 SOMA 没有 MaxPool LayerOp，因此**不要为了它新增专用 MaxPool runtime**。Binary MaxPool 使用现有 Spatial/Conv connectivity 表达：

```text
kernel/stride/padding = 官方 MaxPool 参数
所有 connection weight = 1
threshold = 1
reset = hard
无跨 timestep membrane 保留
```

因为输入为 binary spike，所以该实现严格等价于 MaxPool/OR：同 timestep pooling window 内只要任意 spike 到达，output 就 firing 一次。

连接继续使用紧凑 Spatial Pattern，不展开 neuron-to-neuron connectivity。

后续顺序严格按官方模型：

```text
maxpool1 → proj1_conv+BN → proj2_lif
maxpool2 → proj2_conv+BN → proj3_lif
maxpool3 → proj3_conv+BN → proj4_lif
maxpool4 → proj4_conv+BN
```

### 3. LIF 语义

当前 SOMA LIF 是：

```text
V = leak * V + input + bias
```

而 SpikingJelly `MultiStepLIFNode(tau=2, decay_input=True)` 是：

```text
V = 0.5 * V + 0.5 * X
```

因此新增一个**通用、默认值为 1.0 的 ****`input_scale`**** 配置字段**，使 SOMA LIF 支持：

```text
V = leak * V + input_scale * (input + bias)
```

Spikingformer 普通 LIF 设置：

```text
leak = 0.5
input_scale = 0.5
threshold = 1.0
reset = hard
```

`attn_lif`：

```text
leak = 0.5
input_scale = 0.5
threshold = 0.5
reset = hard
```

默认 `input_scale=1.0`，保证现有 VGG/ResNet/ViT 行为完全不变。

### 4. Spikingformer SSA

严格按照官方：

```text
x
→ proj_lif
→ q/k/v ConvBN
→ q/k/v LIF

KV_t = K_t^T @ V_t
O_t  = Q_t @ KV_t
O_t *= 0.125

→ attn_lif
→ proj ConvBN
→ residual
```

这里**每个 timestep 独立**，禁止使用现有 `IncrementalSpikeMatmul` 的 cumulative shadow：

```text
t1 只使用 Q1/K1/V1
t2 只使用 Q2/K2/V2
...
```

现有 attention runtime 只有 lhs/rhs 两个 operand，而 Spikingformer SSA 需要 Q/K/V 三路，并且 `KV` 中间结果不是 spike，因此不要拆成两个通过 NoC 连接的 spike layer。

新增通用 operator，例如：

```text
operator_type: timestep_spike_attention
```

复用现有 attention timestep-buffer / partition / NoC 基础设施，但每个 partition 当前 timestep 保存 sparse transient：

```text
Q_t
K_t
V_t
```

在 attention Core 内局部完成：

```text
K_t^T @ V_t
Q_t @ KV_t
× 0.125
→ 当前 Core 的 attn_lif
```

`KV_t` 只作为该 Core 当前 timestep 的临时中间量，**不通过 NoC 发送、不跨 timestep 保存**。

扩展现有 `attention_operand` 使该 operator 支持 `q/k/v` 三种 operand；禁止按 layer 名写死。

Q/K/V 继续使用现有 **timestep synchronization + timestep buffer** 对齐。某一路本 timestep 无 spike即按全 0 处理，不新增 completion packet。

统计分别记录：

```text
kv_attention_updates
q_attention_updates
attention_updates = 两者之和
```

并复用现有 attention latency/energy 配置计算实际 scalar accumulation 开销。

### 5. Membrane Shortcut / Residual

Spikingformer：

```text
x = x + attn(x)
x = x + mlp(x)
```

这里 `x` 和 branch output 是 **multi-valued activation，不是 binary spike**。

因此**不能直接复用当前 ViT 的 weighted-identity spike residual**，也禁止把 `x` 错误转换成 spike。

做一个最小、配置驱动的 `membrane_shortcut` / analog state-buffer runtime 扩展：

```text
residual1[t] = x[t] + attn_proj[t]
residual2[t] = residual1[t] + fc2[t]
```

这些 activation 只属于当前 logical timestep，不跨 timestep累积；跨 timestep persistent 的只有各 LIF 自己的 membrane。

优先将 residual state-buffer 与随后消费它的 pre-LIF 做 local/fused mapping，使 projection synaptic accumulation 直接更新对应 local buffer，避免把 dense residual activation伪装成 spike packet在 NoC 上传输。

具体实现必须通用配置驱动，不按 `block0/residual1` 等名字写死。

### 6. 最终 readout

最后一个 Transformer block 的输出与 PyTorch golden trace 对齐后结束 SOMA 仿真；spatial mean、temporal mean 和最终 Linear classification head 在 Python 中执行，仅用于验证最终 prediction，不计入 SOMA hardware metrics。

### 7. Mapping / 验证

遵守现有：

```text
max_neurons=1024
global SpikeQueue
XY routing
timestep synchronization
connection delay
physical Core partition
```

生成类似：

```text
tools/prepare_spikingformer_imagenet_assets.py
tools/compare_spikingformer_reference.py
configs/spikingformer_imagenet_loihi_like.yaml
input/spikingformer_imagenet_input_spike.csv
input/spikingformer_imagenet_weights.npz
compiler/mapping_output/spikingformer_imagenet_mapping.yaml
```

实际运行一个 ImageNet sample 到 `completed=true`。

逐 timestep、逐元素比较：

```text
Tokenizer 后续 LIF
proj_lif
q/k/v LIF
SSA output / attn_lif
attention proj
residual1
MLP LIF
FC outputs
residual2
各 block output
final logits
```

目标 zero mismatch。

最后重跑 VGG regression，要求原有 hardware metrics 不变。

生成 `docs/SPIKINGFORMER_IMAGENET_STATUS.md`，汇报：

* preprocessing 和初始 spike；
* MaxPool、LIF、SSA、membrane shortcut 的实现；
* 新增/修改文件；
* cores / tiles；
* correctness；
* prediction；
* hardware latency / energy  / host latency ；
* packets / NoC hops / synaptic updates / attention updates；
* VGG regression。

不要只生成代码或资产，必须实际跑一个 sample；如果无法 zero mismatch，报告第一个 mismatch 的 node、timestep、index、PyTorch value、SOMA value 和原因。

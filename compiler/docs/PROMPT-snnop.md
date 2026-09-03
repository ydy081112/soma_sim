`snn_op` **先只做 operator-level IR，不碰 mapping、hardware、event execution**。

1. **Dialect 定位与边界**

   * `snn_op` 只描述“这个 SNN 在算什么”，保持 hardware-independent。
   * 输入来自 PyTorch frontend 规范化后的 SNN graph，目前是来自`/home/ydy/compiler/mlir-attention/nir/full_blocks_until_silent_cifar10`，这个文件夹里面是分block来放的，编译出来的mlir直接变成一个.mlir就行
   * `snn_op`的输出未来会 lower 到 `snn_exec`。
   * 不允许出现 `core_id / router / SRAM / buffer / latency` 等硬件信息。
   * 不重新实现普通 `reshape / transpose / add / mul`，需要用的话，直接复用 MLIR 的 `tensor / arith / linalg` 等 dialect。
   * 目前实现的这个vit的ANN转换成的SNN，用的是st-bif这个神经元，关于这个神经元，需要的背景知识在`/home/ydy/compiler/soma_sim/docs/spikeZIP.pdf`
   * 目前的pass的代码风格（不过这次用不到pass，只有一个小的opt应该）以及文件路径可以参照`/home/ydy/compiler/mlir-attention`（文件路径一定要参考他的），但snn dialect的设计千万不要参照他的，里面是一个很toy的版本
   * 目前这个snn_op dialect的设计可以参照`/home/ydy/compiler/soma_sim/docs/(arxiv26)(西班牙)SNN-MLIR An MLIR Dialect for Compiling Neuromorphic SNNs from NIR to Bare-Metal C.pdf`，注意！只是参考，不要照抄他的，更不能长得太相似，应该我是要发论文的。要做出一些自己的特色，我需要写进论文创新点的

2. **V1 Operation Set** 例子，不够可以加：

   * Synaptic：

     * `snn_op.conv2d`
     * `snn_op.linear`
     * `snn_op.qk`
     * `snn_op.av`
     * `snn_op.residual`

   * Spike dataflow：

     * `snn_op.pool`
     * `snn_op.rescale`

   * Neuron：

     * `snn_op.st_bif`
     * `snn_op.lif`
(每一个op都要有输入和输出的形状，比如这样：: memref<200xi8>, memref<256x200xi8> -> memref<256xi32>)

3. **共享 Type / Attribute**

   * 最重要的是 Spike Type，建议最终做成共享类型，例如：

     ```mlir
     !snn.spike<binary>
     !snn.spike<ternary> //这个是2bits
     ```

     这样以后 `snn_exec` 也能直接复用。
   * 至少定义这些 attribute：

     ```text
     spike_encoding = binary | ternary
     neuron_model
     threshold
     tau
     time_dim
     ```
   * Weight 不把完整 tensor 塞进 MLIR，后续 weight packing 再生成 NPZ。

    weight在mlir采用例如如下的方案：添加一个属性来存储参数，比如这个方言：stablehlo.constantop

    %0 = stablehlo.constant {url = "constant_0.bin"} dense<0.000000e+00> : tensor<32000x768xf32>

    但是由于一个文件里面会有conv1,conv2等等，我需要在不同地方引用这个权重文件的不同分块，一个变量很难表达
    所以采用如下方法：

    原理：dense_resource 属性是 MLIR 内置机制的一部分。它允许你将大的二进制数据（权重）从 .mlir 文件的正文中分离出来，作为一个“资源”存储在文件的元数据部分或一个单独的文件中。
    如何实现“分块”：一个外部资源文件（如 .safetensors）本身就包含一个 key -> tensor 的映射。因此，你完全可以将 conv1 和 conv2 的权重作为两个不同的 key 存储在同一个资源文件里。在 MLIR 中，你只需要通过 dense_resource 属性引用对应的 key 即可。这样，.mlir 文件保持小巧，而所有权重都整齐地打包在一个外部文件中。

    对于现在的snn_op，先把权重来源设置为nir，后面在进入模拟器之前的层级再变成npz，然后再把npz引用到里面去

4. **每类 Op 的必要语义**

   * `conv2d`：`kernel / stride / padding / groups / weight / bias`
   * `linear`：`in_features / out_features / weight / bias`
   * `lif`：`threshold / tau / reset_mode / reset_value`
   * `st_bif`：至少保存 `threshold / tr_min / tr_max`，并标记输出是 ternary spike
   * `qk`：`num_heads / head_dim / scale`
   * `av`：`num_heads / head_dim`



5. **State 语义要保留，但暂时不要展开**

   * `snn_op.lif` 本身应该被认为是 stateful op，例如内部逻辑上有：

     ```text
     membrane potential V
     ```
   * `st_bif` 至少有：

     ```text
     membrane V
     spike tracer S
     ```
   * 但是 `snn_op` 阶段不要展开成：

     ```text
     load
     acc
     leak
     threshold
     reset
     ```
   * 这些全部留给后面的：

     ```text
     snn_op → snn_exec
     ```

     lowering。

6. **Verifier 要做**

   * `conv2d` 检查 input/output channel、kernel、groups 合法性。
   * `qk` 检查：

     ```text
     hidden_dim = num_heads × head_dim
     ```
   * `lif` threshold 必须有效。
   * binary neuron 输出必须是 binary spike type。
   * ST-BIF 输出必须允许 ternary spike。

7. **Canonicalization 可以先实现少量关键规则（少量，要非常少量，只做一条rescale都行）**

   * 连续 `rescale` 合并。
   * **暂时不要在这里做 hardware-aware Conv+LIF fusion。**
   * BN folding 在 PyTorch frontend / model normalization 阶段完成，这里也不用做。

8. **建议 MLIR 工程结构**

   ```text
   include/SNNOp/       //每个dialect一个文件夹
   ├── SNNOpDialect.h
   ├── SNNOpDialect.td
   ├── SNNOpOps.td
   ├── SNNOpTypes.td
   └── SNNOpAttrs.td

   lib/SNNOp/          //每个dialect一个文件夹
   ├── SNNOpDialect.cpp
   ├── SNNOpOps.cpp
   └── SNNOpTypes.cpp
   ```

   用 MLIR TableGen 定义 op/type/attribute。

9. **第一阶段验收标准**
   把nir直接lower成snn-op层级的.mlir，然后弄个 `soma-opt` 能 parse / print / verify 一份这样的 MLIR(不做 lowering，只是检查)

   (可以先注册一个soma-opt  --snnop-canonicalize这样)


最核心的设计原则：

> **`snn_op` 保留高层 SNN operator、neuron 和 spike semantics；不要提前展开 state transition，不要展开 synaptic connectivity，也不要引入任何硬件资源。Conv/QK/LIF/ST-BIF 等仍然应该在 IR 中一眼可辨认。**

下一步再实现 `snn_op → snn_exec` 时，才把 `LIF/ST-BIF` 展开成 `state + acc + advance + threshold + reset + fire`，把 `Conv/Linear` 转成 symbolic `connect + spatial pattern`。

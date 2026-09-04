# SNNExec 实现说明

## 完成范围

本阶段在 `compiler/mlir-soma` 中增加了硬件无关的 `snn_exec` dialect，并把已经完成 model-neuron fusion、死输出消除的 SNNOp 图降低到 spike/state 执行 IR。没有加入 spatial pattern、mapping、参数打包、SNNExec 之后的硬件 lowering 或 simulator export。

共享 `snn` dialect 新增两种类型：

- `!snn.voltage<iN|fN>`：一次神经元 population 计算产生的电压值；
- `!snn.state<tensor<...x!snn.voltage<T>>>`：persistent membrane state 声明。它写在 generic region 内，但不是每次事件触发时重新创建的临时值。

`snn_exec` 新增以下 op：

- `snn_exec.generic`：一个 fused neuron population 的执行 region，外部 operands/results 完整继承 DCE 后的 fused SNNOp；
- `snn_exec.state`：persistent membrane，可由 `snn_op.param` bias symbol 初始化；
- `snn_exec.sw`：spike × 参数 weight，或 residual 的 spike × scalar weight；
- `snn_exec.ss`：两组 spike/tracer 的乘积贡献；
- `snn_exec.mul`：norm/affine/rescale 的逐元素贡献；
- `snn_exec.reduce`：pooling 的窗口归约；
- `snn_exec.integrate`：state 与一个或多个 voltage contribution 的累加；
- `snn_exec.fire`：保存 threshold/tracer bounds，并只产生仍然存活的 spike/tracer；
- `snn_exec.yield`：把 region 内结果连接回 generic results。

## Lowering

命令行 pass 为：

```text
--lower-snnop-to-snnexec
```

标准流水线为：

```bash
build/bin/soma-opt \
  --model-neuron-fusion \
  --dead-neuron-out-eliminate \
  --lower-snnop-to-snnexec \
  input.snnop.mlir -o output.snnexec.mlir
```

降低直接按 fused op mnemonic 分类，不依赖 SSA 名字：

- `linear/q/k/v/z/fc/conv2d_stbif` → `state + sw + integrate + fire`；
- `affine/norm/rescale_stbif` → `state + mul + integrate + fire`；
- `qk/qkv_stbif` → `state + ss + integrate + fire`，保留四条 spike/tracer operand 依赖；
- `residual_stbif` → 两个带 `w_main/w_skip` 的 `sw`，再 integrate/fire；
- `pool_stbif` → `state + reduce + integrate + fire`。

module-level `snn_op.param` 在这一层继续作为外部 immutable parameter object；IR 仍不嵌入真实 tensor。generic region 的 population shape 会删除 `time_dim`，而外部结果仍保留完整时间维。voltage、threshold 与 tracer 的数值类型由 NIR metadata 原生 dtype 生成；当前 split-residual-norm block_00 对应 `!snn.voltage<i16>`、`threshold : i16` 和 tracer/bounds `i8`。

## block_00 结果

生成文件为 `compiler/mlir-soma/output/block_00_split_residual_norm.snnexec.mlir`。主要 generic 顺序为：

```text
norm -> q -> k -> v -> qk -> qkv -> z -> residual
     -> norm -> fc -> fc -> residual -> norm
```

合计 13 个 `generic/state/integrate/fire/yield`、10 个 `sw`、3 个 `mul`、2 个 `ss`。Q/K/V/QK 保留 spike+tracer 双结果；普通中间路径只保留实际使用的 spike；末端 norm 保留 spike+tracer 并由 `func.return` 返回。

## 验证

执行并通过：

```bash
cmake --build compiler/mlir-soma/build -j2
ctest --test-dir compiler/mlir-soma/build --output-on-failure
# 9/9 passed

compiler/mlir-soma/build/bin/soma-opt --verify-each \
  compiler/mlir-soma/test/snnexec_ops.mlir -o /tmp/snnexec-roundtrip.mlir
compiler/mlir-soma/build/bin/soma-opt --verify-each \
  /tmp/snnexec-roundtrip.mlir -o /dev/null
```

真实 block_00 也完成 importer、SNNOp verify、fusion、DCE、SNNExec lowering，以及最终 parse/verify/print round-trip。旧的 `vit_blocks.snnop.mlir` 已用新的 split-residual-norm 六 block 目录重新生成并通过 verifier；共 106 个 parameter symbol，106 个唯一名称。

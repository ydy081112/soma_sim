# SNNOp 到 Core-level SNNArch mapping

## 范围

本阶段只建立 fused SNNOp、physical Core partition 和 Core-level NoC edge。没有引入 SNNExec mapping、router path、buffer/port/arbiter、microarchitecture、weight packing 或 simulator export。

命令：

```bash
compiler/mlir-soma/build/bin/soma-opt --verify-each \
  '--snnop-core-mapping=hardware-file=compiler/mlir-soma/output/hardware.core.mlir' \
  compiler/mlir-soma/output/block_00_split_residual_norm.fused.dce.mlir \
  -o compiler/mlir-soma/output/block_00_split_residual_norm.core.mlir
```

硬件 symbol 也可以预先和 SNNOp 放在同一 module，此时不传 `hardware-file`。默认引用 `@noc0`、`@standard_core`；可用 pass option `network`、`core-type` 改名。

## Dialect 扩展

`snn_arch.core_type` 新增可选 `neuron_model`；mapping 输出把所选 Core type 标为 `"st_bif"`。

新增 model-aware Core op：

- `conv2d_core`、`linear_core`
- `q_core`、`k_core`、`v_core`、`z_core`、`fc_core`
- `affine_core`、`norm_core`
- `qk_core`、`qkv_core`、`residual_core`
- `rescale_core`、`pool_core`

每个 Core op 原样继承 fused op 的结果 type 和语义属性，并增加 `core_type/core_id/coord/partition_id/partition_offset/partition_size`。operand 是 Variadic tensor；四个同长度数组逐项描述输入来源：

- `source_operand`：原 fused op 的逻辑 operand 序号；
- `source_partition`：producer partition 序号，函数输入用 `-1`；
- `source_offset`、`source_size`：producer population 中的扁平区间。

没有 `partition_join`。shape-preserving 的 norm/affine/rescale/residual 只连接 offset 区间相交的 producer partition；dense/shape-changing 与 QK/QKV 运算保守连接对应逻辑 operand 的全部 producer partitions。最终 `func.return` 直接返回末端 spike/tracer 的所有 partitions，并同步更新函数结果类型。

NoC 新增 `send_router`、`recv_router`。同 Core edge 直接使用 SSA；跨 Core 的每一个 partition value 依次经过 send（只存 source coord）和 recv（只存 destination coord），两者保持原 tensor type，不记录中间 router。

## Partition 与 placement

1. 按函数中的 SSA/支配顺序处理 fused op。
2. 输出 tensor 删除 `time_dim` 后的静态 shape 乘积是 neuron population。
3. 以 `neuron_capacity` 连续切分，最后一片可以小于 capacity。
4. 对每片从 `core_id=0` 开始 first-fit：优先选择剩余容量足够的已有 Core，否则分配新 Core。
5. `coord = [core_id % dimensions[0], core_id / dimensions[0]]`；超过 mesh 总容量立即报错。

## 真实 block_00 结果

- mapping 前：`output/block_00_split_residual_norm.fused.dce.mlir`，13 个 fused op。
- 硬件输入：`output/hardware.core.mlir`，`neuron_capacity=1024`、mesh `[128,256]`。
- mapping 后：`output/block_00_split_residual_norm.core.mlir`。
- 输出包含 128 个 model-aware Core partitions；末端 norm 的 8 个 spike partitions 和 8 个 tracer partitions 被显式 return。
- 不含 `snn_arch.partition_join`；QK/QKV 保留 spike/tracer 的四个逻辑输入，residual 保留两路输入及 `w_main/w_skip`。

## 验证

```bash
cmake --build compiler/mlir-soma/build --target soma-opt -j2
ctest --test-dir compiler/mlir-soma/build --output-on-failure
compiler/mlir-soma/build/bin/soma-opt --verify-each \
  compiler/mlir-soma/output/block_00_split_residual_norm.core.mlir \
  -o /tmp/block_core_roundtrip.mlir
```

CTest 覆盖 hardware symbol parse/print、capacity partition、first-fit reuse、placement metadata、同 Core 无 NoC、跨 Core router pair、Q/K/V/QK/QKV/residual spike/tracer SSA，以及 mesh overflow 负例。

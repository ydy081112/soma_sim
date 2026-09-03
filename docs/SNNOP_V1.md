# snn_op V1 完成说明

我在 `compiler/mlir-soma/` 建立了独立的 MLIR 子工程，目标严格限定为硬件无关的 SNN operator IR。

- 共享类型为 `!snn.spike<binary>` 和 `!snn.spike<ternary>`；神经元输出因此不再退化为裸 `i8`。
- 实现了 `snn_op.conv2d`、`linear`、`qk`、`av`、`residual`、`pool`、`rescale`、`lif`、`st_bif`。Op 均在文本中保留输入/输出 tensor 形状与高层语义。
- `lif` 约束 binary spike 输出，`st_bif` 约束 ternary spike 输出；还检查 Conv group/channel、QK 的 head/hidden 关系和关键 neuron 参数。
- 新增 `soma-opt`，可 parse、print、verify；`--snnop-canonicalize` 会合并连续 `rescale`，不做硬件感知 fusion。
- 新增 NIR 1.0.8 HDF5 导入器。指定目录的 6 个真实 ViT blocks 已归并导出到 `compiler/mlir-soma/output/vit_blocks.snnop.mlir`；其权重仅以 `nir://...#/node/...` key 引用，不把大 tensor 写入 MLIR。

验证已经完成：工程可由本机 MLIR 23 构建；smoke IR 的 rescale 合并通过；归并后的 6-block IR 由 `soma-opt --verify-each` 通过。

当前没有实现 `snn_op -> snn_exec`、NPZ weight packing、mapping、hardware 或 event execution；这些属于下一阶段。

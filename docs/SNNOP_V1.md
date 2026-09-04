# snn_op V1 完成说明

## 参数 symbol 化更新

`snn_op.param` 是 module-level symbol，用 `source` URI 指向 NIR 中的不可变
weight/bias，不嵌入 tensor 内容。`snn_op.linear` 和 `snn_op.conv2d` 通过
`weight = @...`（以及可选的 `bias = @...`）引用这些参数；verifier 会解析 symbol
并确认 weight/bias 的 kind。NIR 目录导入为每个 block/node 加前缀，保证合并后 symbol
全局唯一。

split-residual-norm NIR 的通道仿射节点以 `snn_op.norm` 表达：它引用 weight/bias
symbols，并带 `axis = -1` 与 `time_dim = 0`；输入和输出 shape 必须一致。该图还将
Q/K/V/O projection 导出为 `x_wq/x_wk/x_wv/z_wo`，MLP 导出为 `fc`，attention value
聚合导出为 `qkv`；generic `linear/affine/av` 保持兼容而不用于该 block。

我在 `compiler/mlir-soma/` 建立了独立的 MLIR 子工程，目标严格限定为硬件无关的 SNN operator IR。

- 共享类型为 `!snn.spike<binary>` 和 `!snn.spike<ternary>`；神经元输出因此不再退化为裸 `i8`。
- 实现了 `snn_op.param`、`conv2d`、`linear`、`affine`、`x_wq/x_wk/x_wv/z_wo/fc/norm`、`qk`、`av/qkv`、`residual`、`pool`、`rescale`、`lif`、`st_bif`。Op 均在文本中保留输入/输出 tensor 形状与高层语义。
- `lif` 约束 binary spike 输出，`st_bif` 约束 ternary spike 输出；还检查 Conv group/channel、QK 的 head/hidden 关系和关键 neuron 参数。
- 新增 `soma-opt`，可 parse、print、verify；`--snnop-canonicalize` 会合并连续 `rescale`，不做硬件感知 fusion。
- 新增 NIR 1.0.8 HDF5 导入器。指定目录的 6 个真实 ViT blocks 已归并导出到 `compiler/mlir-soma/output/vit_blocks.snnop.mlir`；其参数以 module-level `snn_op.param` 的 `nir://...#/node/...` source 引用，计算 op 只使用 parameter symbol，不把大 tensor 写入 MLIR。

验证已经完成：工程可由本机 MLIR 23 构建；CTest 3/3 覆盖有效 weight/bias、可省略 bias、缺失 parameter 和错误 kind；smoke IR 的 rescale 合并通过；单一 NIR block 与归并后的 6-block IR 均由 `soma-opt --verify-each` 通过。

当前没有实现 `snn_op -> snn_exec`、NPZ weight packing、mapping、hardware 或 event execution；这些属于下一阶段。

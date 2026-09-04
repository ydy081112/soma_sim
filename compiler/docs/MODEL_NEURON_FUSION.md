# SNNOp Model-Neuron Fusion

## 完成内容

融合前的 `snn_op.st_bif` 保持普通 SSA 传递：它接受并返回完全相同的普通数值 tensor，携带 `threshold`、`tr_min`、`tr_max` 和 `time_dim`。因此下一个 model op 使用 `st_bif` 的输出；`!snn.spike<ternary>` 只在融合后出现。

新增 `--model-neuron-fusion`。pass 通过 Operation mnemonic 和直接 def-use 边识别 `model -> st_bif`，不检查 SSA 名称或 NIR node 字符串；它合并两侧属性、创建 fused op、重连下游 consumer，随后删除原始 model op 与注释 op。

已定义并注册的 fused op：

- `conv2d_stbif`、`linear_stbif`、`q_stbif`、`k_stbif`、`v_stbif`、`z_stbif`、`fc_stbif`
- `affine_stbif`、`norm_stbif`、`qk_stbif`、`qkv_stbif`、`residual_stbif`
- `rescale_stbif`、`pool_stbif`

fusion 首先为 fused op 产生同一逻辑 shape 的 spike 与 tracer：spike 为 `tensor<...x!snn.spike<ternary>>`，tracer 的数值 element type 取 NIR `metadata/output` dtype（当前 block_00 为 `i8`），而非固定 `i32`。QK/QKV 分别接收两组 `(spike, tracer)`，从而在 IR 中显式表达 `Q×K` 和 `S×V` 的双通路依赖。residual 使用 `w_main`、`w_skip`；NIR importer 将 `metadata/shifter1/shifter2` 转换为 `1 << shift`。

`--dead-neuron-out-eliminate` 随后按 def-use 独立删除死掉的 spike 或 tracer result；若两个 result 都没有 use，则删除整个 fused op。函数末端的 `return` 被 fusion 扩展为最后一组 `(spike, tracer)`，故二者都是活跃输出。

## block_00 结果

输入 NIR：`block_00_until_silent_t30_cls_free_split_residual_norm.nir`。

融合后的主要次序（由该 block 的实际 graph 决定）为：

```text
norm_stbif → q_stbif/k_stbif/v_stbif → qk_stbif → qkv_stbif
→ z_stbif → residual_stbif → norm_stbif → fc_stbif → fc_stbif
→ residual_stbif → norm_stbif
```

其中 QK 的 IR 形状为：

```mlir
%6, %7 = snn_op.qk_stbif %0, %1, %2, %3 { ... }
  : (...) -> (tensor<...x!snn.spike<ternary>>, tensor<...xi8>)
```

fused op 的每个 result 都是独立 SSA group，因此 `soma-opt` 始终打印为
`%0, %1 = ...` 和后续 `%0`/`%1` 引用，不使用 MLIR 默认的 `%0:2`、`%0#0` 压缩形式。

产物：

- 融合前：[block_00_split_residual_norm.snnop.mlir](../mlir-soma/output/block_00_split_residual_norm.snnop.mlir)
- 融合后：[block_00_split_residual_norm.fused.mlir](../mlir-soma/output/block_00_split_residual_norm.fused.mlir)
- 死输出清理后：[block_00_split_residual_norm.fused.dce.mlir](../mlir-soma/output/block_00_split_residual_norm.fused.dce.mlir)

## 验证

```bash
cmake --build compiler/mlir-soma/build --target soma-opt -j2
ctest --test-dir compiler/mlir-soma/build --output-on-failure
python3 compiler/mlir-soma/tools/nir-to-snnop.py ...block_00...nir --output ...snnop.mlir
soma-opt --verify-each --model-neuron-fusion ...snnop.mlir > ...fused.mlir
soma-opt --verify-each --dead-neuron-out-eliminate ...fused.mlir > ...fused.dce.mlir
```

结果：构建成功；目标 NIR 的导入、fusion、死输出清理后的 print/parse/verify 均通过。该阶段没有加入 SNNExec、spatial pattern、mapping、hardware lowering、NPZ/weight packing。

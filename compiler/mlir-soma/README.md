# MLIR SOMA：SNNOp 与 SNNExec

`snn_op` 是硬件无关的 operator-level SNN IR；它不包含 mapping、router、SRAM、buffer 或 latency。

构建：`cmake -S . -B build -G Ninja -DMLIR_DIR=/home/ydy/compiler/install/lib/cmake/mlir && cmake --build build -j2`

导入指定 NIR blocks：

`conda run -n sim_snn python tools/nir-to-snnop.py --input /home/ydy/compiler/mlir-attention/nir/full_blocks_until_silent --output output/vit_blocks.snnop.mlir`。`--input` 也可给单一 `.nir` block。

检查/规范化：`build/bin/soma-opt --verify-each --snnop-canonicalize output/block00.snnop.mlir`

导入器先生成 module-level `snn_op.param @...`，其中保存 NIR URI、kind、shape 和 dtype；`linear/conv2d` 再用 `weight = @...`（可选 `bias = @...`）引用它。大 tensor 不进入文本 MLIR；weight packing 后可替换 parameter 的 source。

split-residual-norm NIR 的 `norm1_IF`、`norm1_IF_next`、`norm2_IF` 显式导为 `snn_op.norm`，引用 [128] weight/bias；其余当前支持的 node name 同样通过显式表分派，未知节点会报错。该图使用 `x_wq/x_wk/x_wv/z_wo`、`fc`、`norm`、`qk`、`qkv`、`residual`；通用 `linear/affine/av` 保持兼容。

将 fused SNNOp 降低到硬件无关的 spike/state 执行 IR：

`build/bin/soma-opt --model-neuron-fusion --dead-neuron-out-eliminate --lower-snnop-to-snnexec input.mlir -o output.mlir`

SNNExec 使用 `generic/state/sw/ss/mul/reduce/integrate/fire/yield`，以及共享的 `!snn.voltage<T>`、`!snn.state<...>` 类型。详细说明与验证结果见 `compiler/docs/SNNEXEC.md`。

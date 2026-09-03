## [进行中]

## [已完成]

- 2026-09-03：完成 `mlir-soma` 的 `snn_op` V1。新增共享 `!snn.spike<binary|ternary>`、TableGen 定义的 `conv2d/linear/qk/av/residual/pool/rescale/lif/st_bif`、关键 shape/spike/neuron verifier 和 `--snnop-canonicalize`（连续 rescale 合并）。新增 NIR 1.0.8 HDF5 block 导入器，权重保留为 `nir://...#/node/...` resource key，不嵌入 tensor。指定目录的 6 个 ViT blocks 已归并输出为单一 `mlir-soma/output/vit_blocks.snnop.mlir`，其中包含 `linear → st_bif`、`qk → st_bif`、`av → st_bif`。验证：`cmake -S compiler/mlir-soma -B compiler/mlir-soma/build -G Ninja -DMLIR_DIR=/home/ydy/compiler/install/lib/cmake/mlir`、`cmake --build compiler/mlir-soma/build --target soma-opt -j2`、`soma-opt --verify-each --snnop-canonicalize test/snnop_smoke.mlir`、以及真实 blocks 导入后的 `soma-opt --verify-each` 均成功。未覆盖：仅实现 V1 import/verify，未实现 snn_exec lowering、weight packing 与 mapping/hardware/event execution。

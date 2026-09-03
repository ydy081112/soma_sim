# MLIR SOMA：snn_op V1

`snn_op` 是硬件无关的 operator-level SNN IR；它不包含 mapping、router、SRAM、buffer 或 latency。

构建：`cmake -S . -B build -G Ninja -DMLIR_DIR=/home/ydy/compiler/install/lib/cmake/mlir && cmake --build build -j2`

导入指定 NIR blocks：

`conda run -n sim_snn python tools/nir-to-snnop.py --input /home/ydy/compiler/mlir-attention/nir/full_blocks_until_silent --output output/vit_blocks.snnop.mlir`。`--input` 也可给单一 `.nir` block。

检查/规范化：`build/bin/soma-opt --verify-each --snnop-canonicalize output/block00.snnop.mlir`

权重引用形如 `nir://block_00...nir#/node/nodes/q_IF/metadata/weight`，因此大 tensor 不进入文本 MLIR；weight packing 后可替换为 NPZ resource。

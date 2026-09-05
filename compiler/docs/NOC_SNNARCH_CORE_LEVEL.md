# NoC / SNNArch Core-Level IR

本阶段新增两个硬件侧 MLIR dialect，只表达 module-level 的最小 core-level 声明，不包含 core 实例、placement、mapping、静态 route、router/buffer/SRAM/soma/synapse 微架构或 simulator export。

## IR

```mlir
module {
  noc.network @noc0 {
    topology = "mesh",
    dimensions = [8, 8],
    hop_latency = 2ns,
    routing = "xy"
  }
  snn_arch.core_type @standard_core {
    neuron_capacity = 1024
  }
}
```

- `noc.network` 和 `snn_arch.core_type` 都是直接位于 module 的 MLIR symbol，可被后续 placement/mapping op 用 `@name` 引用。
- NoC 第一版限定 `mesh`、二维正 dimensions 和 `xy`；`hop_latency` 使用 `ps/ns/us/ms/s` 的带单位自定义时间属性。文本保留单位，例如 `4.1ns`，不在该层换算为 ps。
- Core 第一版只有正整数 `neuron_capacity`；刻意不把 `synapse_sram_bytes` 误解释为 synapse entry capacity。

## YAML 前端

使用 `tools/hardware-yaml-to-arch.py`：

```bash
conda run -n sim_snn python tools/hardware-yaml-to-arch.py \
  --input ../../arch/hardware.yaml \
  --output output/hardware.core.mlir
build/bin/soma-opt --verify-each output/hardware.core.mlir -o /dev/null
```

字段映射：

| YAML | Core-level IR |
| --- | --- |
| `architecture.noc.topology` | `topology` |
| `architecture.noc.rows`, `cols` | `dimensions` |
| `architecture.noc.link.hardware_latency` | `hop_latency` |
| `architecture.noc.routing` | `routing` |
| `architecture.core.max_neurons` | `neuron_capacity` |

所有现有 `arch/*.yaml` 已将 `routing` 更新为 `xy`；这与当前 simulator 的确定性 XY 路径一致。方向相关 link latency、buffer 和 SRAM 等更细字段仍由后续 microarchitecture layer 表达。

## 验证

新增正例、dimensions/routing/neuron capacity 负例和 YAML 前端 CTest。完成时执行：

```bash
cmake --build compiler/mlir-soma/build --target soma-opt -j2
ctest --test-dir compiler/mlir-soma/build --output-on-failure
# 14/14 passed
```

`arch/hardware.yaml` 的生成结果位于 `compiler/mlir-soma/output/hardware.core.mlir`，为 128×256 mesh、4.1ns hop、XY routing 和 1024-neuron core。

include "mlir/IR/OpBase.td"

def SNNOpDialect : Dialect {
  let name = "snn_op";
  let cppNamespace = "::snn_op";
  let summary = "硬件无关的 SNN operator-level IR";
  let description = [{保留神经元、spike 与高层突触算子语义；不含 mapping、存储或时序。}];
}

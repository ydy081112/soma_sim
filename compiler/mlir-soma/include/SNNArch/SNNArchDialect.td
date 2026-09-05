include "mlir/IR/OpBase.td"

def SNNArchDialect : Dialect {
  let name = "snn_arch";
  let cppNamespace = "::snn_arch";
  let summary = "core-level SNN architecture resource declaration";
  let description = [{仅描述可被后续 placement 引用的 Core type。}];
}

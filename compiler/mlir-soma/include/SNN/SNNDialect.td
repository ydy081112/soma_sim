include "mlir/IR/OpBase.td"

def SNNDialect : Dialect {
  let name = "snn";
  let cppNamespace = "::snn";
  let summary = "共享的 SNN 值语义";
  let description = [{仅承载跨层共享的 spike type，不描述任何硬件资源。}];
}

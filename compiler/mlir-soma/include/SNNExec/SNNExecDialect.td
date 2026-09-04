include "mlir/IR/DialectBase.td"

def SNNExecDialect : Dialect {
  let name = "snn_exec";
  let cppNamespace = "::snn_exec";
  let summary = "面向 spike/state 的硬件无关执行 IR";
  let description = [{
    SNNExec 展开模型级算子为 spike contribution、persistent state、
    integration 与 firing，但不包含 mapping 或具体硬件资源。
  }];
}

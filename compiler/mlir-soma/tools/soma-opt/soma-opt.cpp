#include "SNN/SNNDialect.h"
#include "SNNOp/SNNOpDialect.h"
#include "SNNOp/SNNOpPasses.h"
#include "SNNExec/SNNExecDialect.h"
#include "SNNExec/SNNExecPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::registerAllPasses(); snn_op::registerSNNOpPasses(); snn_exec::registerSNNExecPasses();
  mlir::DialectRegistry registry;
  registry.insert<snn::SNNDialect, snn_op::SNNOpDialect, snn_exec::SNNExecDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();
  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "SOMA SNN operator IR optimizer\n", registry));
}

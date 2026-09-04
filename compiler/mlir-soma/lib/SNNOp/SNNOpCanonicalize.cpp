#include "SNNOp/SNNOpOps.h"
#include "SNNOp/SNNOpPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace snn_op {
namespace {
struct MergeRescale final : mlir::OpRewritePattern<RescaleOp> {
  using OpRewritePattern::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(RescaleOp op, mlir::PatternRewriter &rewriter) const override {
    auto prior = op.getInput().getDefiningOp<RescaleOp>();
    if (!prior || prior.getTimeDim() != op.getTimeDim()) return mlir::failure();
    rewriter.replaceOpWithNewOp<RescaleOp>(op, op.getOutput().getType(), prior.getInput(),
      rewriter.getF64FloatAttr(prior.getScale().convertToDouble() * op.getScale().convertToDouble()), op.getTimeDimAttr());
    return mlir::success();
  }
};
struct CanonicalizePass final : mlir::PassWrapper<CanonicalizePass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizePass)
  llvm::StringRef getArgument() const final { return "snnop-canonicalize"; }
  llvm::StringRef getDescription() const final { return "merge adjacent hardware-independent snn_op.rescale operations"; }
  void runOnOperation() override { mlir::RewritePatternSet patterns(&getContext()); patterns.add<MergeRescale>(&getContext()); if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure(); }
};
} // namespace
void registerSNNOpPasses() {
  mlir::PassRegistration<CanonicalizePass>();
  registerModelNeuronFusionPass();
  registerDeadNeuronOutEliminatePass();
}
} // namespace snn_op

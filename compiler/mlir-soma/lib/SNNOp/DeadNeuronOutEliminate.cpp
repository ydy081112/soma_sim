#include "SNNOp/SNNOpPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace snn_op {
namespace {

static bool isFusedNeuronOp(Operation *op) {
  StringRef mnemonic = op->getName().getStringRef();
  return mnemonic.starts_with("snn_op.") && mnemonic.ends_with("_stbif");
}

/// Removes only values that are provably dead in the SSA graph. A fused op
/// whose spike and tracer are both dead has no observable effect and is erased.
struct DeadNeuronOutEliminatePass final
    : PassWrapper<DeadNeuronOutEliminatePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeadNeuronOutEliminatePass)

  StringRef getArgument() const final { return "dead-neuron-out-eliminate"; }
  StringRef getDescription() const final {
    return "eliminate unused spike/tracer results and dead fused neuron operations";
  }

  void runOnOperation() override {
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (isFusedNeuronOp(op)) candidates.push_back(op);
    });

    OpBuilder builder(&getContext());
    for (Operation *op : llvm::reverse(candidates)) {
      // An earlier erased producer may have made this operation disappear from
      // the list only when it was nested; fused ops are regionless, so keep a
      // simple defensive attachment check.
      if (!op->getBlock()) continue;

      SmallVector<unsigned> liveIndices;
      for (auto [index, result] : llvm::enumerate(op->getResults()))
        if (!result.use_empty()) liveIndices.push_back(index);
      if (liveIndices.size() == op->getNumResults()) continue;
      if (liveIndices.empty()) {
        op->erase();
        continue;
      }

      OperationState state(op->getLoc(), op->getName());
      state.addOperands(op->getOperands());
      state.addAttributes(op->getAttrs());
      SmallVector<Type> resultTypes;
      for (unsigned index : liveIndices)
        resultTypes.push_back(op->getResult(index).getType());
      state.addTypes(resultTypes);
      builder.setInsertionPoint(op);
      Operation *replacement = builder.create(state);
      for (auto [newIndex, oldIndex] : llvm::enumerate(liveIndices))
        op->getResult(oldIndex).replaceAllUsesWith(replacement->getResult(newIndex));
      op->erase();
    }
  }
};

} // namespace

void registerDeadNeuronOutEliminatePass() {
  PassRegistration<DeadNeuronOutEliminatePass>();
}

} // namespace snn_op

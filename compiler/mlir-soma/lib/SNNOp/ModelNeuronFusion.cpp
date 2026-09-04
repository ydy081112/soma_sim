#include "SNN/SNNDialect.h"
#include "SNNOp/SNNOpOps.h"
#include "SNNOp/SNNOpPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

namespace snn_op {
namespace {

/// The pair is deliberately identified by the operation name and its def-use
/// edge, never by an SSA spelling or frontend node name.
struct FusionCandidate {
  Operation *model;
  Operation *neuron;
  StringRef fusedMnemonic;
};

struct SpikeTracerValues {
  Value spike;
  Value tracer;
};

static StringRef fusedMnemonicFor(Operation *op) {
  return llvm::StringSwitch<StringRef>(op->getName().getStringRef())
      .Case("snn_op.conv2d", "snn_op.conv2d_stbif")
      .Case("snn_op.linear", "snn_op.linear_stbif")
      .Case("snn_op.x_wq", "snn_op.q_stbif")
      .Case("snn_op.x_wk", "snn_op.k_stbif")
      .Case("snn_op.x_wv", "snn_op.v_stbif")
      .Case("snn_op.z_wo", "snn_op.z_stbif")
      .Case("snn_op.fc", "snn_op.fc_stbif")
      .Case("snn_op.affine", "snn_op.affine_stbif")
      .Case("snn_op.norm", "snn_op.norm_stbif")
      .Case("snn_op.qk", "snn_op.qk_stbif")
      .Case("snn_op.av", "snn_op.qkv_stbif")
      .Case("snn_op.qkv", "snn_op.qkv_stbif")
      .Case("snn_op.residual", "snn_op.residual_stbif")
      .Case("snn_op.rescale", "snn_op.rescale_stbif")
      .Case("snn_op.pool", "snn_op.pool_stbif")
      .Default("");
}

static bool needsSpikeTracerOperands(StringRef mnemonic) {
  return mnemonic == "snn_op.qk_stbif" || mnemonic == "snn_op.qkv_stbif";
}

static void appendMergedAttributes(Operation *model, Operation *neuron,
                                   OperationState &state) {
  llvm::DenseSet<StringAttr> seen;
  SmallVector<NamedAttribute> attributes;
  auto append = [&](Operation *op) {
    for (NamedAttribute attribute : op->getAttrs())
      if (seen.insert(attribute.getName()).second)
        attributes.push_back(attribute);
  };
  append(model);
  // time_dim exists on both operations and has model-op precedence.  All other
  // neuron properties are copied verbatim into the fused operation.
  append(neuron);
  state.addAttributes(attributes);
}

struct ModelNeuronFusionPass final
    : PassWrapper<ModelNeuronFusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModelNeuronFusionPass)

  StringRef getArgument() const final { return "model-neuron-fusion"; }
  StringRef getDescription() const final {
    return "fuse snn_op model operations with following snn_op.st_bif annotations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<FusionCandidate> candidates;
    llvm::DenseMap<Operation *, unsigned> candidateByModel;
    bool failed = false;

    module.walk([&](STBIFOp neuron) {
      Operation *model = neuron.getInput().getDefiningOp();
      if (!model) {
        neuron.emitOpError("requires its input to be produced by a fusible snn_op model operation");
        failed = true;
        return;
      }
      StringRef mnemonic = fusedMnemonicFor(model);
      if (mnemonic.empty()) {
        neuron.emitOpError() << "does not support fusion with producer '"
                             << model->getName().getStringRef() << "'";
        failed = true;
        return;
      }
      if (model->getNumResults() != 1 || neuron.getInput() != model->getResult(0)) {
        neuron.emitOpError("requires a direct single-result producer-consumer edge");
        failed = true;
        return;
      }
      if (!dyn_cast<RankedTensorType>(model->getResult(0).getType())) {
        neuron.emitOpError("requires a ranked tensor model result");
        failed = true;
        return;
      }
      auto modelTimeDim = dyn_cast_or_null<IntegerAttr>(model->getAttr("time_dim"));
      if (!modelTimeDim || modelTimeDim != neuron.getTimeDimAttr()) {
        neuron.emitOpError("requires the paired model operation to have the same time_dim");
        failed = true;
        return;
      }
      if (!candidateByModel.try_emplace(model, candidates.size()).second) {
        neuron.emitOpError("has a model producer already paired with another snn_op.st_bif");
        failed = true;
        return;
      }
      candidates.push_back({model, neuron.getOperation(), mnemonic});
    });
    if (failed) {
      signalPassFailure();
      return;
    }

    llvm::DenseSet<Operation *> pairedNeurons;
    for (const FusionCandidate &candidate : candidates) {
      pairedNeurons.insert(candidate.neuron);
    }
    // The restored standalone st_bif is the sole user of its model result.
    // Its own SSA result is what carries the graph to later model operations.
    for (const FusionCandidate &candidate : candidates) {
      for (Operation *user : candidate.model->getResult(0).getUsers()) {
        if (!pairedNeurons.contains(user)) {
          candidate.model->emitOpError("has a non-st_bif result user during model-neuron fusion");
          signalPassFailure();
          return;
        }
      }
    }

    OpBuilder builder(&getContext());
    llvm::DenseMap<Value, SpikeTracerValues> replacements;

    for (const FusionCandidate &candidate : candidates) {
      SmallVector<Value> operands;
      if (needsSpikeTracerOperands(candidate.fusedMnemonic)) {
        if (candidate.model->getNumOperands() != 2) {
          candidate.model->emitOpError("requires exactly two operands for spike/tracer attention fusion");
          signalPassFailure();
          return;
        }
        for (Value input : candidate.model->getOperands()) {
          auto it = replacements.find(input);
          if (it == replacements.end()) {
            candidate.model->emitOpError("requires both attention inputs to come from fused spike/tracer producers");
            signalPassFailure();
            return;
          }
          operands.push_back(it->second.spike);
          operands.push_back(it->second.tracer);
        }
      } else {
        for (Value input : candidate.model->getOperands()) {
          auto it = replacements.find(input);
          operands.push_back(it == replacements.end() ? input : it->second.spike);
        }
      }

      auto neuron = cast<STBIFOp>(candidate.neuron);
      auto modelType = cast<RankedTensorType>(neuron.getOutput().getType());
      Type spikeElement = snn::SpikeType::get(&getContext(), snn::SpikeEncoding::Ternary);
      Type spikeType = RankedTensorType::get(modelType.getShape(), spikeElement);
      // The tracer carries the NIR node's output numeric representation rather
      // than an importer-invented i32 type.
      Type tracerType = neuron.getOutput().getType();

      OperationState state(candidate.model->getLoc(), candidate.fusedMnemonic);
      state.addOperands(operands);
      appendMergedAttributes(candidate.model, candidate.neuron, state);
      state.addTypes({spikeType, tracerType});
      builder.setInsertionPoint(candidate.model);
      Operation *fused = builder.create(state);
      replacements[neuron.getOutput()] = {fused->getResult(0), fused->getResult(1)};
    }

    // The importer returns its terminal standalone st_bif output. Rewriting it
    // to both explicit fused outputs makes both terminal values live for the
    // subsequent dead-neuron-output pass.
    bool returnRewriteFailed = false;
    module.walk([&](func::FuncOp function) {
      SmallVector<Type> rewrittenResultTypes;
      bool changedFunction = false;
      function.walk([&](func::ReturnOp ret) {
        SmallVector<Value> operands;
        bool changedReturn = false;
        for (Value operand : ret.getOperands()) {
          auto it = replacements.find(operand);
          if (it == replacements.end()) {
            operands.push_back(operand);
            continue;
          }
          operands.push_back(it->second.spike);
          operands.push_back(it->second.tracer);
          changedReturn = changedFunction = true;
        }
        if (!changedReturn) return;
        SmallVector<Type> types;
        for (Value operand : operands) types.push_back(operand.getType());
        if (!rewrittenResultTypes.empty() && rewrittenResultTypes != types) {
          ret.emitOpError("requires all returns to have the same rewritten result types");
          returnRewriteFailed = true;
          return;
        }
        rewrittenResultTypes = std::move(types);
        ret->setOperands(operands);
      });
      if (changedFunction && !returnRewriteFailed)
        function.setType(FunctionType::get(&getContext(), function.getArgumentTypes(),
                                           rewrittenResultTypes));
    });
    if (returnRewriteFailed) {
      signalPassFailure();
      return;
    }

    // Erase in reverse graph order. Each standalone st_bif must go first so
    // the paired model result has no remaining consumer when it is erased.
    for (auto it = candidates.rbegin(), end = candidates.rend(); it != end; ++it) {
      it->neuron->erase();
      it->model->erase();
    }
  }
};

} // namespace

void registerModelNeuronFusionPass() {
  PassRegistration<ModelNeuronFusionPass>();
}

} // namespace snn_op

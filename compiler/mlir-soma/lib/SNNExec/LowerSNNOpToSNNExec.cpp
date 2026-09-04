#include "SNN/SNNDialect.h"
#include "SNNExec/SNNExecOps.h"
#include "SNNExec/SNNExecPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace snn_exec {
namespace {

enum class LoweringKind { SW, Mul, SS, Residual, Reduce, Unsupported };

static LoweringKind loweringKind(StringRef mnemonic) {
  return llvm::StringSwitch<LoweringKind>(mnemonic)
      .Cases("snn_op.conv2d_stbif", "snn_op.linear_stbif",
             "snn_op.q_stbif", "snn_op.k_stbif", "snn_op.v_stbif",
             "snn_op.z_stbif", "snn_op.fc_stbif", LoweringKind::SW)
      .Cases("snn_op.affine_stbif", "snn_op.norm_stbif",
             "snn_op.rescale_stbif", LoweringKind::Mul)
      .Cases("snn_op.qk_stbif", "snn_op.qkv_stbif", LoweringKind::SS)
      .Case("snn_op.residual_stbif", LoweringKind::Residual)
      .Case("snn_op.pool_stbif", LoweringKind::Reduce)
      .Default(LoweringKind::Unsupported);
}

static FailureOr<Type> parseScalarType(MLIRContext *context, StringRef spelling) {
  if (spelling.consume_front("i")) {
    unsigned width = 0;
    if (!spelling.empty() && !spelling.getAsInteger(10, width) && width > 0)
      return IntegerType::get(context, width);
    return failure();
  }
  if (spelling == "f16") return Float16Type::get(context);
  if (spelling == "f32") return Float32Type::get(context);
  if (spelling == "f64") return Float64Type::get(context);
  return failure();
}

static FailureOr<RankedTensorType> stripTimeDimension(Type type,
                                                       int64_t timeDim,
                                                       Type elementType = {}) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || timeDim < 0 || timeDim >= tensor.getRank()) return failure();
  SmallVector<int64_t> shape(tensor.getShape());
  shape.erase(shape.begin() + timeDim);
  return RankedTensorType::get(shape,
                               elementType ? elementType : tensor.getElementType());
}

static void copyAttributes(Operation *source, OperationState &target,
                           ArrayRef<StringRef> names) {
  for (StringRef name : names)
    if (Attribute attribute = source->getAttr(name))
      target.addAttribute(name, attribute);
}

static Operation *createOperation(OpBuilder &builder, Location location,
                                  StringRef mnemonic, ValueRange operands,
                                  TypeRange results,
                                  ArrayRef<NamedAttribute> attributes = {}) {
  OperationState state(location, mnemonic);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attributes);
  return builder.create(state);
}

static LogicalResult createWeightedDelta(Operation *source, OpBuilder &builder,
                                         StringRef mnemonic, Value input,
                                         Type voltageType, Attribute weight,
                                         ArrayRef<StringRef> copiedAttributes,
                                         Value &delta) {
  if (!weight)
    return source->emitOpError("requires a weight for SNNExec lowering");
  OperationState state(source->getLoc(), mnemonic);
  state.addOperands(input);
  state.addTypes(voltageType);
  state.addAttribute("weight", weight);
  copyAttributes(source, state, copiedAttributes);
  delta = builder.create(state)->getResult(0);
  return success();
}

static LogicalResult lowerOne(Operation *source, OpBuilder &outerBuilder) {
  LoweringKind kind = loweringKind(source->getName().getStringRef());
  if (kind == LoweringKind::Unsupported)
    return source->emitOpError("is not a supported fused SNNOp");

  auto timeDimAttr = source->getAttrOfType<IntegerAttr>("time_dim");
  auto voltageDtypeAttr = source->getAttrOfType<StringAttr>("voltage_dtype");
  if (!timeDimAttr || !voltageDtypeAttr)
    return source->emitOpError("requires time_dim and voltage_dtype");
  FailureOr<Type> voltageElement =
      parseScalarType(source->getContext(), voltageDtypeAttr.getValue());
  if (failed(voltageElement))
    return source->emitOpError("has an unsupported voltage_dtype");
  if (source->getNumResults() == 0 || source->getNumResults() > 2)
    return source->emitOpError("requires one or two live results after DCE");

  FailureOr<RankedTensorType> voltageTensor = stripTimeDimension(
      source->getResult(0).getType(), timeDimAttr.getInt(),
      snn::VoltageType::get(source->getContext(), *voltageElement));
  if (failed(voltageTensor))
    return source->emitOpError("cannot derive the per-timestep voltage shape");

  SmallVector<Type> innerResultTypes;
  for (Value result : source->getResults()) {
    FailureOr<RankedTensorType> inner =
        stripTimeDimension(result.getType(), timeDimAttr.getInt());
    if (failed(inner))
      return source->emitOpError("cannot derive an inner result type");
    innerResultTypes.push_back(*inner);
  }

  OperationState genericState(source->getLoc(), GenericOp::getOperationName());
  genericState.addOperands(source->getOperands());
  genericState.addTypes(source->getResultTypes());
  genericState.addAttribute("time_dim", timeDimAttr);
  genericState.addRegion();
  outerBuilder.setInsertionPoint(source);
  Operation *generic = outerBuilder.create(genericState);
  Region &body = generic->getRegion(0);
  body.push_back(new Block());
  OpBuilder builder = OpBuilder::atBlockBegin(&body.front());

  OperationState stateState(source->getLoc(), StateOp::getOperationName());
  if (Attribute bias = source->getAttr("bias")) stateState.addAttribute("init", bias);
  Type stateType = snn::StateType::get(source->getContext(), *voltageTensor);
  stateState.addTypes(stateType);
  Value membrane = builder.create(stateState)->getResult(0);

  SmallVector<Value> deltas;
  switch (kind) {
  case LoweringKind::SW: {
    Value delta;
    if (failed(createWeightedDelta(
            source, builder, SWOp::getOperationName(), source->getOperand(0),
            *voltageTensor, source->getAttr("weight"),
            {"in_features", "out_features", "kernel", "stride", "padding",
             "groups"}, delta)))
      return failure();
    StringRef name = source->getName().getStringRef();
    if ((name == "snn_op.q_stbif" || name == "snn_op.k_stbif" ||
         name == "snn_op.v_stbif") &&
        voltageTensor->getRank() >= 3) {
      auto heads = builder.getI64IntegerAttr(
          voltageTensor->getDimSize(voltageTensor->getRank() - 3));
      auto headDim = builder.getI64IntegerAttr(
          voltageTensor->getDimSize(voltageTensor->getRank() - 1));
      delta.getDefiningOp()->setAttr("num_heads", heads);
      delta.getDefiningOp()->setAttr("head_dim", headDim);
    }
    deltas.push_back(delta);
    break;
  }
  case LoweringKind::Mul: {
    Attribute weight = source->getAttr("weight");
    if (!weight) weight = source->getAttr("scale");
    Value delta;
    if (failed(createWeightedDelta(source, builder, MulOp::getOperationName(),
                                   source->getOperand(0), *voltageTensor,
                                   weight, {"axis"}, delta)))
      return failure();
    deltas.push_back(delta);
    break;
  }
  case LoweringKind::SS: {
    if (source->getNumOperands() != 4)
      return source->emitOpError("requires four spike/tracer operands");
    OperationState ssState(source->getLoc(), SSOp::getOperationName());
    ssState.addOperands(source->getOperands());
    ssState.addTypes(*voltageTensor);
    copyAttributes(source, ssState, {"num_heads", "head_dim", "scale"});
    deltas.push_back(builder.create(ssState)->getResult(0));
    break;
  }
  case LoweringKind::Residual: {
    if (source->getNumOperands() != 2)
      return source->emitOpError("requires two residual inputs");
    for (auto [input, attrName] :
         llvm::zip(source->getOperands(), ArrayRef<StringRef>{"w_main", "w_skip"})) {
      Value delta;
      if (failed(createWeightedDelta(source, builder, SWOp::getOperationName(),
                                     input, *voltageTensor,
                                     source->getAttr(attrName), {}, delta)))
        return failure();
      deltas.push_back(delta);
    }
    break;
  }
  case LoweringKind::Reduce: {
    OperationState reduceState(source->getLoc(), ReduceOp::getOperationName());
    reduceState.addOperands(source->getOperand(0));
    reduceState.addTypes(*voltageTensor);
    copyAttributes(source, reduceState, {"kind", "kernel", "stride"});
    deltas.push_back(builder.create(reduceState)->getResult(0));
    break;
  }
  case LoweringKind::Unsupported:
    llvm_unreachable("checked above");
  }

  SmallVector<Value> integrateOperands{membrane};
  llvm::append_range(integrateOperands, deltas);
  Operation *integrate = createOperation(builder, source->getLoc(),
                                         IntegrateOp::getOperationName(),
                                         integrateOperands, {*voltageTensor});

  OperationState fireState(source->getLoc(), FireOp::getOperationName());
  fireState.addOperands(integrate->getResult(0));
  fireState.addTypes(innerResultTypes);
  copyAttributes(source, fireState, {"threshold", "tr_min", "tr_max"});
  Operation *fire = builder.create(fireState);
  createOperation(builder, source->getLoc(), YieldOp::getOperationName(),
                  fire->getResults(), {});

  source->replaceAllUsesWith(generic->getResults());
  source->erase();
  return success();
}

struct LowerSNNOpToSNNExecPass final
    : PassWrapper<LowerSNNOpToSNNExecPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerSNNOpToSNNExecPass)

  StringRef getArgument() const final { return "lower-snnop-to-snnexec"; }
  StringRef getDescription() const final {
    return "lower fused SNNOp populations to spike/state SNNExec regions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SNNExecDialect, snn::SNNDialect>();
  }

  void runOnOperation() override {
    SmallVector<Operation *> fused;
    getOperation().walk([&](Operation *op) {
      if (loweringKind(op->getName().getStringRef()) != LoweringKind::Unsupported)
        fused.push_back(op);
    });
    OpBuilder builder(&getContext());
    for (Operation *op : fused) {
      if (failed(lowerOne(op, builder))) {
        signalPassFailure();
        return;
      }
    }

    bool illegal = false;
    getOperation().walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (name.starts_with("snn_op.") && name != "snn_op.param") {
        op->emitOpError("must be fused and lowered before entering SNNExec");
        illegal = true;
      }
    });
    if (illegal) signalPassFailure();
  }
};

} // namespace

void registerLowerSNNOpToSNNExecPass() {
  PassRegistration<LowerSNNOpToSNNExecPass>();
}

} // namespace snn_exec

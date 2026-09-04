#include "SNNExec/SNNExecDialect.h"
#include "SNNExec/SNNExecOps.h"
#include "SNNOp/SNNOpOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace snn_exec;

#include "SNNExec/SNNExecDialect.cpp.inc"

void SNNExecDialect::initialize() {
  addOperations<GenericOp, StateOp, SWOp, SSOp, MulOp, ReduceOp, IntegrateOp,
                FireOp, YieldOp>();
}

static RankedTensorType ranked(Type type) {
  return dyn_cast<RankedTensorType>(type);
}

static bool isSpikeTensor(Type type) {
  auto tensor = ranked(type);
  auto spike = tensor ? dyn_cast<snn::SpikeType>(tensor.getElementType())
                      : snn::SpikeType();
  return spike && spike.getEncoding() == snn::SpikeEncoding::Ternary;
}

static bool isNumericTensor(Type type) {
  auto tensor = ranked(type);
  return tensor && isa<IntegerType, FloatType>(tensor.getElementType());
}

static bool isVoltageTensor(Type type) {
  auto tensor = ranked(type);
  return tensor && isa<snn::VoltageType>(tensor.getElementType());
}

static bool sameShape(Type lhsType, Type rhsType) {
  auto lhs = ranked(lhsType), rhs = ranked(rhsType);
  return lhs && rhs && lhs.getShape() == rhs.getShape();
}

static FailureOr<RankedTensorType> stripTimeDimension(Type type,
                                                       int64_t timeDim) {
  auto tensor = ranked(type);
  if (!tensor || timeDim < 0 || timeDim >= tensor.getRank()) return failure();
  SmallVector<int64_t> shape(tensor.getShape());
  shape.erase(shape.begin() + timeDim);
  return RankedTensorType::get(shape, tensor.getElementType());
}

static LogicalResult verifyParameter(Operation *op, FlatSymbolRefAttr ref,
                                     StringRef kind) {
  auto parameter = SymbolTable::lookupNearestSymbolFrom<snn_op::ParamOp>(op, ref);
  if (!parameter)
    return op->emitOpError() << "references unknown snn_op.param " << ref;
  if (parameter.getKind() != kind)
    return op->emitOpError() << "requires kind=\"" << kind << "\" parameter, but "
                             << ref << " has kind=\"" << parameter.getKind() << "\"";
  return success();
}

static std::optional<double> numericValue(Attribute attribute) {
  if (auto integer = dyn_cast<IntegerAttr>(attribute))
    return integer.getValue().getSExtValue();
  if (auto floating = dyn_cast<FloatAttr>(attribute))
    return floating.getValueAsDouble();
  return std::nullopt;
}

static void nameResults(Operation *op, OpAsmSetValueNameFn setNameFn) {
  for (Value result : op->getResults()) setNameFn(result, "");
}

void GenericOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  nameResults(getOperation(), setNameFn);
}

void FireOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  nameResults(getOperation(), setNameFn);
}

LogicalResult GenericOp::verify() {
  if (getInputs().empty() || getOutputs().empty() || getOutputs().size() > 2)
    return emitOpError("requires inputs and one or two live outputs");
  const int64_t timeDim = getTimeDimAttr().getInt();
  if (timeDim < 0)
    return emitOpError("requires non-negative time_dim");
  if (getBody().empty()) return emitOpError("requires a non-empty body");

  Block &block = getBody().front();
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield) return emitOpError("body must terminate with snn_exec.yield");
  if (yield.getValues().size() != getOutputs().size())
    return emitOpError("yield/result count mismatch");

  for (auto [external, internal] : llvm::zip(getOutputs(), yield.getValues())) {
    FailureOr<RankedTensorType> expected =
        stripTimeDimension(external.getType(), timeDim);
    if (failed(expected) || *expected != internal.getType())
      return emitOpError("yield type must equal the corresponding result type with time_dim removed");
  }

  SmallVector<StateOp> states;
  block.walk([&](StateOp state) { states.push_back(state); });
  if (states.size() != 1)
    return emitOpError("requires exactly one persistent snn_exec.state");
  auto stateType = dyn_cast<snn::StateType>(states.front().getState().getType());
  auto payload = stateType ? ranked(stateType.getPayloadType()) : RankedTensorType();
  FailureOr<RankedTensorType> population =
      stripTimeDimension(getOutputs().front().getType(), timeDim);
  if (!payload || failed(population) || payload.getShape() != (*population).getShape())
    return emitOpError("state shape must equal the output population shape without time_dim");
  return success();
}

LogicalResult StateOp::verify() {
  if (!isa_and_nonnull<GenericOp>((*this)->getParentOp()))
    return emitOpError("must be nested directly in snn_exec.generic");
  auto stateType = dyn_cast<snn::StateType>(getState().getType());
  auto payload = stateType ? ranked(stateType.getPayloadType()) : RankedTensorType();
  if (!payload || !isa<snn::VoltageType>(payload.getElementType()))
    return emitOpError("requires !snn.state<tensor<...x!snn.voltage<...>>>");
  if (getInitAttr() && failed(verifyParameter(*this, getInitAttr(), "bias")))
    return failure();
  return success();
}

static LogicalResult verifyWeightedContribution(Operation *op, Value input,
                                                Attribute weight, Value output) {
  if (!isSpikeTensor(input.getType()))
    return op->emitOpError("requires a ternary spike tensor input");
  if (!isVoltageTensor(output.getType()))
    return op->emitOpError("requires a tensor of !snn.voltage output");
  if (auto ref = dyn_cast<FlatSymbolRefAttr>(weight))
    return verifyParameter(op, ref, "weight");
  auto scalar = numericValue(weight);
  if (!scalar || *scalar == 0.0)
    return op->emitOpError("weight must be a weight symbol or non-zero numeric scalar");
  return success();
}

LogicalResult SWOp::verify() {
  return verifyWeightedContribution(*this, getInput(), getWeight(), getOutput());
}

LogicalResult MulOp::verify() {
  return verifyWeightedContribution(*this, getInput(), getWeight(), getOutput());
}

LogicalResult SSOp::verify() {
  if (!isSpikeTensor(getLhsSpike().getType()) ||
      !isSpikeTensor(getRhsSpike().getType()))
    return emitOpError("requires ternary spike operands");
  if (!isNumericTensor(getLhsTracer().getType()) ||
      !isNumericTensor(getRhsTracer().getType()))
    return emitOpError("requires numeric tracer operands");
  if (!sameShape(getLhsSpike().getType(), getLhsTracer().getType()) ||
      !sameShape(getRhsSpike().getType(), getRhsTracer().getType()))
    return emitOpError("requires matching spike/tracer shapes");
  if (!isVoltageTensor(getOutput().getType()))
    return emitOpError("requires a tensor of !snn.voltage output");
  auto heads = (*this)->getAttrOfType<IntegerAttr>("num_heads");
  auto dimension = (*this)->getAttrOfType<IntegerAttr>("head_dim");
  if (!heads || !dimension || heads.getInt() <= 0 || dimension.getInt() <= 0)
    return emitOpError("requires positive num_heads and head_dim attributes");
  return success();
}

LogicalResult ReduceOp::verify() {
  if (!isSpikeTensor(getInput().getType()) || !isVoltageTensor(getOutput().getType()))
    return emitOpError("requires ternary spike input and voltage output");
  if (getKind() != "avg" && getKind() != "max")
    return emitOpError("kind must be 'avg' or 'max'");
  if (getKernel().empty() || getStride().empty())
    return emitOpError("requires non-empty kernel and stride");
  return success();
}

LogicalResult IntegrateOp::verify() {
  auto stateType = dyn_cast<snn::StateType>(getState().getType());
  if (!stateType || getDeltas().empty())
    return emitOpError("requires a state and at least one voltage contribution");
  Type payload = stateType.getPayloadType();
  if (getOutput().getType() != payload)
    return emitOpError("result type must equal the state payload type");
  for (Value delta : getDeltas())
    if (delta.getType() != payload)
      return emitOpError("all voltage contributions must equal the state payload type");
  return success();
}

LogicalResult FireOp::verify() {
  if (!isVoltageTensor(getInput().getType()))
    return emitOpError("requires a tensor of !snn.voltage input");
  auto threshold = numericValue(getThreshold());
  auto trMin = numericValue(getTrMin());
  auto trMax = numericValue(getTrMax());
  if (!threshold || *threshold <= 0.0 || !trMin || !trMax || *trMin > *trMax)
    return emitOpError("requires positive numeric threshold and ordered numeric tracer bounds");
  if (getOutputs().empty() || getOutputs().size() > 2)
    return emitOpError("requires one or two live neuron outputs");
  bool sawSpike = false, sawTracer = false;
  for (Value output : getOutputs()) {
    if (!sameShape(getInput().getType(), output.getType()))
      return emitOpError("output shapes must match the voltage population");
    if (isSpikeTensor(output.getType())) {
      if (sawSpike) return emitOpError("cannot have multiple spike outputs");
      sawSpike = true;
    } else if (isNumericTensor(output.getType())) {
      if (sawTracer) return emitOpError("cannot have multiple tracer outputs");
      sawTracer = true;
    } else {
      return emitOpError("outputs must be ternary spike or numeric tracer tensors");
    }
  }
  return success();
}

LogicalResult YieldOp::verify() {
  if (!isa_and_nonnull<GenericOp>((*this)->getParentOp()))
    return emitOpError("must terminate snn_exec.generic");
  return success();
}

#define GET_OP_CLASSES
#include "SNNExec/SNNExecOps.cpp.inc"

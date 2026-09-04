#include "SNNOp/SNNOpDialect.h"
#include "SNNOp/SNNOpOps.h"
#include "SNN/SNNDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace snn_op;

#include "SNNOp/SNNOpDialect.cpp.inc"

void SNNOpDialect::initialize() {
  addOperations<ParamOp, Conv2DOp, LinearOp, XWQOp, XWKOp, XWVOp, ZWOOp, FCOp,
                AffineOp, NormOp, QKOp, AVOp, QKVOp, ResidualOp, PoolOp,
                RescaleOp, LIFOp, STBIFOp, Conv2DSTBIFOp, LinearSTBIFOp,
                QSTBIFOp, KSTBIFOp, VSTBIFOp, ZSTBIFOp, FCSTBIFOp,
                AffineSTBIFOp, NormSTBIFOp, QKSTBIFOp, QKVSTBIFOp,
                ResidualSTBIFOp, RescaleSTBIFOp, PoolSTBIFOp>();
}

static bool compatible(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}
static RankedTensorType tensorType(Value value) { return dyn_cast<RankedTensorType>(value.getType()); }
static LogicalResult sameShape(Operation *op, Value a, Value b) {
  auto lhs = tensorType(a), rhs = tensorType(b);
  if (!lhs || !rhs || lhs.getRank() != rhs.getRank()) return op->emitOpError("requires equal-rank tensor types");
  for (int64_t i = 0; i < lhs.getRank(); ++i)
    if (!compatible(lhs.getDimSize(i), rhs.getDimSize(i))) return op->emitOpError("requires matching tensor shapes");
  return success();
}
static bool encodingIs(mlir::Type type, snn::SpikeEncoding encoding) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  auto spike = tensor ? dyn_cast<snn::SpikeType>(tensor.getElementType()) : snn::SpikeType();
  return spike && spike.getEncoding() == encoding;
}

LogicalResult ParamOp::verify() {
  if (getKind() != "weight" && getKind() != "bias")
    return emitOpError("kind must be 'weight' or 'bias'");
  if (getSource().empty() || getDtype().empty())
    return emitOpError("requires non-empty source and dtype");
  // 空 shape 表示标量参数；实际 tensor 维度必须为正。
  for (int64_t dimension : getShape())
    if (dimension <= 0) return emitOpError("shape dimensions must be positive");
  return success();
}

static LogicalResult verifyParameterRef(Operation *op, FlatSymbolRefAttr ref,
                                        StringRef requiredKind, StringRef role) {
  auto parameter = SymbolTable::lookupNearestSymbolFrom<ParamOp>(op, ref);
  if (!parameter)
    return op->emitOpError() << role << " references unknown snn_op.param " << ref;
  if (parameter.getKind() != requiredKind)
    return op->emitOpError() << role << " must reference kind=\"" << requiredKind
                             << "\", but " << ref << " has kind=\""
                             << parameter.getKind() << "\"";
  return success();
}

static LogicalResult verifyLinearLike(Operation *op, Value input, Value output,
                                      int64_t inFeatures, int64_t outFeatures,
                                      FlatSymbolRefAttr weight,
                                      FlatSymbolRefAttr bias) {
  auto in = tensorType(input), out = tensorType(output);
  if (!in || !out || in.getRank() < 1 || out.getRank() < 1)
    return op->emitOpError("requires ranked tensors");
  if (inFeatures <= 0 || outFeatures <= 0)
    return op->emitOpError("requires positive feature counts");
  if (in.getRank() == out.getRank() && !ShapedType::isDynamic(in.getDimSize(in.getRank() - 1)) &&
      in.getDimSize(in.getRank() - 1) != inFeatures)
    return op->emitOpError("input final dimension must equal in_features when rank is unchanged");
  if (out.getRank() == in.getRank() &&
      !compatible(out.getDimSize(out.getRank() - 1), outFeatures))
    return op->emitOpError("output final dimension must equal out_features when rank is unchanged");
  if (failed(verifyParameterRef(op, weight, "weight", "weight"))) return failure();
  if (bias && failed(verifyParameterRef(op, bias, "bias", "bias"))) return failure();
  return success();
}

static LogicalResult verifyAffineLike(Operation *op, Value input, Value output,
                                      FlatSymbolRefAttr weight, FlatSymbolRefAttr bias) {
  if (failed(sameShape(op, input, output))) return failure();
  if (failed(verifyParameterRef(op, weight, "weight", "weight"))) return failure();
  if (failed(verifyParameterRef(op, bias, "bias", "bias"))) return failure();
  return success();
}

static bool isNumericElement(Type type) {
  return isa<IntegerType, FloatType>(type);
}

/// Fused operations initially have both results. The dead-neuron-output pass
/// may retain either one, so identify their roles from their element types.
static LogicalResult verifyFusedResults(Operation *op) {
  if (op->getNumResults() == 0 || op->getNumResults() > 2)
    return op->emitOpError("requires one or two fused neuron results");
  Value spike, tracer;
  for (Value result : op->getResults()) {
    if (encodingIs(result.getType(), snn::SpikeEncoding::Ternary)) {
      if (spike) return op->emitOpError("cannot have more than one spike result");
      spike = result;
      continue;
    }
    auto tensor = tensorType(result);
    if (!tensor || !isNumericElement(tensor.getElementType()))
      return op->emitOpError("tracer result must use a numeric element type");
    if (tracer) return op->emitOpError("cannot have more than one tracer result");
    tracer = result;
  }
  if (spike && tracer && failed(sameShape(op, spike, tracer))) return failure();
  return success();
}

static Value fusedModelResult(Operation *op) {
  for (Value result : op->getResults())
    if (encodingIs(result.getType(), snn::SpikeEncoding::Ternary)) return result;
  return op->getResult(0);
}

// Keep every fused result as its own SSA group. This intentionally uses empty
// names so the standard printer allocates ordinary numeric IDs (`%0, %1`)
// instead of compressing adjacent results as `%0:2`.
static void nameFusedResults(Operation *op, OpAsmSetValueNameFn setNameFn) {
  for (Value result : op->getResults()) setNameFn(result, "");
}

#define DEFINE_FUSED_RESULT_NAMES(Op)                                          \
  void Op::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {                   \
    nameFusedResults(getOperation(), setNameFn);                                \
  }
DEFINE_FUSED_RESULT_NAMES(Conv2DSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(LinearSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(QSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(KSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(VSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(ZSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(FCSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(AffineSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(NormSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(QKSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(QKVSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(ResidualSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(RescaleSTBIFOp)
DEFINE_FUSED_RESULT_NAMES(PoolSTBIFOp)
#undef DEFINE_FUSED_RESULT_NAMES

static LogicalResult verifySpikeTracerOperands(Operation *op, Value spike,
                                                Value tracer, StringRef role) {
  auto spikeType = tensorType(spike), tracerType = tensorType(tracer);
  if (!spikeType || !tracerType || spikeType.getRank() != tracerType.getRank())
    return op->emitOpError() << role << " spike/tracer operands must have equal rank";
  for (int64_t i = 0; i < spikeType.getRank(); ++i)
    if (!compatible(spikeType.getDimSize(i), tracerType.getDimSize(i)))
      return op->emitOpError() << role << " spike/tracer operands must have matching shapes";
  if (!encodingIs(spike.getType(), snn::SpikeEncoding::Ternary))
    return op->emitOpError() << role << " spike operand must use !snn.spike<ternary>";
  if (!isNumericElement(tracerType.getElementType()))
    return op->emitOpError() << role << " tracer operand must use a numeric element type";
  return success();
}

static std::optional<double> numericAttrValue(Attribute attribute) {
  if (auto integer = dyn_cast<IntegerAttr>(attribute))
    return integer.getValue().getSExtValue();
  if (auto floating = dyn_cast<FloatAttr>(attribute))
    return floating.getValueAsDouble();
  return std::nullopt;
}

static bool validScalarDtype(StringRef dtype) {
  if (dtype.consume_front("i")) {
    unsigned width = 0;
    return !dtype.empty() && !dtype.getAsInteger(10, width) && width > 0;
  }
  return dtype == "f16" || dtype == "f32" || dtype == "f64";
}

static LogicalResult verifySTBIFAttrs(Operation *op, Attribute threshold,
                                      Attribute trMin, Attribute trMax,
                                      StringRef voltageDtype) {
  auto thresholdValue = numericAttrValue(threshold);
  auto minValue = numericAttrValue(trMin);
  auto maxValue = numericAttrValue(trMax);
  if (!thresholdValue || !minValue || !maxValue || *thresholdValue <= 0.0 ||
      *minValue > *maxValue)
    return op->emitOpError("requires positive threshold and tr_min <= tr_max");
  if (!validScalarDtype(voltageDtype))
    return op->emitOpError("requires a valid voltage_dtype such as i16 or f32");
  return success();
}

LogicalResult Conv2DOp::verify() {
  auto in = tensorType(getInput()), out = tensorType(getOutput());
  if (!in || !out || in.getRank() != 4 || out.getRank() != 4) return emitOpError("requires rank-4 [time, channel, height, width] input/output");
  if (getGroups() <= 0 || getKernel().empty() || getStride().empty() || getPadding().empty()) return emitOpError("requires positive groups and non-empty kernel/stride/padding");
  if (!ShapedType::isDynamic(in.getDimSize(1)) && in.getDimSize(1) % getGroups() != 0) return emitOpError("requires input channels divisible by groups");
  if (!ShapedType::isDynamic(out.getDimSize(1)) && out.getDimSize(1) % getGroups() != 0) return emitOpError("requires output channels divisible by groups");
  if (failed(verifyParameterRef(*this, getWeightAttr(), "weight", "weight"))) return failure();
  if (getBiasAttr() && failed(verifyParameterRef(*this, getBiasAttr(), "bias", "bias"))) return failure();
  return success();
}
LogicalResult LinearOp::verify() {
  return verifyLinearLike(*this, getInput(), getOutput(), getInFeatures(), getOutFeatures(),
                          getWeightAttr(), getBiasAttr());
}
LogicalResult XWQOp::verify() { return verifyLinearLike(*this, getInput(), getOutput(), getInFeatures(), getOutFeatures(), getWeightAttr(), getBiasAttr()); }
LogicalResult XWKOp::verify() { return verifyLinearLike(*this, getInput(), getOutput(), getInFeatures(), getOutFeatures(), getWeightAttr(), getBiasAttr()); }
LogicalResult XWVOp::verify() { return verifyLinearLike(*this, getInput(), getOutput(), getInFeatures(), getOutFeatures(), getWeightAttr(), getBiasAttr()); }
LogicalResult ZWOOp::verify() { return verifyLinearLike(*this, getInput(), getOutput(), getInFeatures(), getOutFeatures(), getWeightAttr(), getBiasAttr()); }
LogicalResult FCOp::verify() { return verifyLinearLike(*this, getInput(), getOutput(), getInFeatures(), getOutFeatures(), getWeightAttr(), getBiasAttr()); }
LogicalResult AffineOp::verify() {
  return verifyAffineLike(*this, getInput(), getOutput(), getWeightAttr(), getBiasAttr());
}
LogicalResult NormOp::verify() { return verifyAffineLike(*this, getInput(), getOutput(), getWeightAttr(), getBiasAttr()); }
LogicalResult QKOp::verify() {
  auto q = tensorType(getQuery()), k = tensorType(getKey()), out = tensorType(getOutput());
  if (!q || !k || !out || q.getRank() < 1 || k.getRank() < 1 || out.getRank() < 1) return emitOpError("requires ranked tensors");
  if (getNumHeads() <= 0 || getHeadDim() <= 0 || getScale().convertToDouble() <= 0.0) return emitOpError("requires positive num_heads, head_dim, and scale");
  int64_t hidden = getNumHeads() * getHeadDim();
  auto validHidden = [&](RankedTensorType type) {
    // 支持 frontend 已显式拆分 head 的 [..., head, head_dim]，也支持扁平 hidden。
    if (type.getRank() >= 3 && compatible(type.getDimSize(type.getRank()-3), getNumHeads()))
      return compatible(type.getDimSize(type.getRank()-1), getHeadDim());
    return compatible(type.getDimSize(type.getRank()-1), hidden);
  };
  if (!validHidden(q) || !validHidden(k)) return emitOpError("requires hidden_dim = num_heads * head_dim");
  return success();
}
LogicalResult AVOp::verify() {
  if (getNumHeads() <= 0 || getHeadDim() <= 0) return emitOpError("requires positive num_heads and head_dim");
  return success();
}
LogicalResult QKVOp::verify() {
  if (getNumHeads() <= 0 || getHeadDim() <= 0) return emitOpError("requires positive num_heads and head_dim");
  return success();
}
LogicalResult ResidualOp::verify() {
  if (getWMain() <= 0 || getWSkip() <= 0)
    return emitOpError("requires positive w_main and w_skip");
  if (failed(sameShape(*this, getLhs(), getRhs())) ||
      failed(sameShape(*this, getLhs(), getOutput()))) return failure();
  return success();
}
LogicalResult PoolOp::verify() { if (getKind() != "avg" && getKind() != "max") return emitOpError("kind must be avg or max"); return success(); }
LogicalResult RescaleOp::verify() { if (getScale().convertToDouble() == 0.0) return emitOpError("scale must be non-zero"); return success(); }
LogicalResult LIFOp::verify() {
  if (getThreshold().convertToDouble() <= 0.0 || getTau().convertToDouble() <= 0.0) return emitOpError("requires positive threshold and tau");
  if (!encodingIs(getOutput().getType(), snn::SpikeEncoding::Binary)) return emitOpError("binary LIF output must use !snn.spike<binary>");
  return success();
}
LogicalResult STBIFOp::verify() {
  if (failed(sameShape(*this, getInput(), getOutput())) ||
      getInput().getType() != getOutput().getType())
    return emitOpError("requires identical input and output tensor types");
  return verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(),
                          getVoltageDtype());
}

static LogicalResult verifyLinearSTBIF(Operation *op, Value input, int64_t inFeatures,
                                       int64_t outFeatures, FlatSymbolRefAttr weight,
                                       FlatSymbolRefAttr bias, Attribute threshold,
                                       Attribute trMin, Attribute trMax,
                                       StringRef voltageDtype) {
  if (failed(verifyFusedResults(op)) ||
      failed(verifyLinearLike(op, input, fusedModelResult(op), inFeatures, outFeatures, weight, bias)) ||
      failed(verifySTBIFAttrs(op, threshold, trMin, trMax, voltageDtype))) return failure();
  return success();
}
#define VERIFY_LINEAR_STBIF(OP) \
  LogicalResult OP::verify() { return verifyLinearSTBIF(*this, getInput(), getInFeatures(), getOutFeatures(), getWeightAttr(), getBiasAttr(), getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()); }
VERIFY_LINEAR_STBIF(LinearSTBIFOp)
VERIFY_LINEAR_STBIF(QSTBIFOp)
VERIFY_LINEAR_STBIF(KSTBIFOp)
VERIFY_LINEAR_STBIF(VSTBIFOp)
VERIFY_LINEAR_STBIF(ZSTBIFOp)
VERIFY_LINEAR_STBIF(FCSTBIFOp)
#undef VERIFY_LINEAR_STBIF

static LogicalResult verifyAffineSTBIF(Operation *op, Value input,
                                       FlatSymbolRefAttr weight,
                                       FlatSymbolRefAttr bias, Attribute threshold,
                                       Attribute trMin, Attribute trMax,
                                       StringRef voltageDtype) {
  if (failed(verifyFusedResults(op)) ||
      failed(verifyAffineLike(op, input, fusedModelResult(op), weight, bias)) ||
      failed(verifySTBIFAttrs(op, threshold, trMin, trMax, voltageDtype))) return failure();
  return success();
}
LogicalResult AffineSTBIFOp::verify() { return verifyAffineSTBIF(*this, getInput(), getWeightAttr(), getBiasAttr(), getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()); }
LogicalResult NormSTBIFOp::verify() { return verifyAffineSTBIF(*this, getInput(), getWeightAttr(), getBiasAttr(), getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()); }

LogicalResult Conv2DSTBIFOp::verify() {
  if (failed(verifyFusedResults(*this))) return failure();
  auto in = tensorType(getInput()), output = tensorType(fusedModelResult(*this));
  if (!in || !output || in.getRank() != 4 || output.getRank() != 4)
    return emitOpError("requires rank-4 input and result");
  if (getGroups() <= 0 || getKernel().empty() || getStride().empty() || getPadding().empty())
    return emitOpError("requires positive groups and non-empty kernel/stride/padding");
  if (failed(verifyParameterRef(*this, getWeightAttr(), "weight", "weight")) ||
      (getBiasAttr() && failed(verifyParameterRef(*this, getBiasAttr(), "bias", "bias"))) ||
      failed(verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()))) return failure();
  return success();
}

LogicalResult QKSTBIFOp::verify() {
  if (getNumHeads() <= 0 || getHeadDim() <= 0 || getScale().convertToDouble() <= 0.0)
    return emitOpError("requires positive num_heads, head_dim, and scale");
  if (failed(verifySpikeTracerOperands(*this, getQuerySpike(), getQueryTracer(), "query")) ||
      failed(verifySpikeTracerOperands(*this, getKeySpike(), getKeyTracer(), "key")) ||
      failed(verifyFusedResults(*this)) ||
      failed(verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()))) return failure();
  return success();
}
LogicalResult QKVSTBIFOp::verify() {
  if (getNumHeads() <= 0 || getHeadDim() <= 0) return emitOpError("requires positive num_heads and head_dim");
  if (failed(verifySpikeTracerOperands(*this, getAttentionSpike(), getAttentionTracer(), "attention")) ||
      failed(verifySpikeTracerOperands(*this, getValueSpike(), getValueTracer(), "value")) ||
      failed(verifyFusedResults(*this)) ||
      failed(verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()))) return failure();
  return success();
}
LogicalResult ResidualSTBIFOp::verify() {
  if (failed(verifyFusedResults(*this))) return failure();
  if (getWMain() <= 0 || getWSkip() <= 0)
    return emitOpError("requires positive w_main and w_skip");
  if (!encodingIs(getMainSpike().getType(), snn::SpikeEncoding::Ternary) ||
      !encodingIs(getSkipSpike().getType(), snn::SpikeEncoding::Ternary))
    return emitOpError("requires ternary spike operands");
  if (failed(sameShape(*this, getMainSpike(), getSkipSpike())) ||
      failed(sameShape(*this, getMainSpike(), fusedModelResult(*this))) ||
      failed(verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()))) return failure();
  return success();
}
LogicalResult RescaleSTBIFOp::verify() {
  if (getScale().convertToDouble() == 0.0 || failed(verifyFusedResults(*this)) ||
      failed(verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()))) return failure();
  return success();
}
LogicalResult PoolSTBIFOp::verify() {
  if ((getKind() != "avg" && getKind() != "max") || failed(verifyFusedResults(*this)) ||
      failed(verifySTBIFAttrs(*this, getThreshold(), getTrMin(), getTrMax(), getVoltageDtype()))) return failure();
  return success();
}

#define GET_OP_CLASSES
#include "SNNOp/SNNOpOps.cpp.inc"

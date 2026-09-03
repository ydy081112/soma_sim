#include "SNNOp/SNNOpDialect.h"
#include "SNNOp/SNNOpOps.h"
#include "SNN/SNNDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace snn_op;

#include "SNNOp/SNNOpDialect.cpp.inc"

void SNNOpDialect::initialize() {
  addOperations<Conv2DOp, LinearOp, QKOp, AVOp, ResidualOp, PoolOp,
                RescaleOp, LIFOp, STBIFOp>();
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

LogicalResult Conv2DOp::verify() {
  auto in = tensorType(getInput()), out = tensorType(getOutput());
  if (!in || !out || in.getRank() != 4 || out.getRank() != 4) return emitOpError("requires rank-4 [time, channel, height, width] input/output");
  if (getGroups() <= 0 || getKernel().empty() || getStride().empty() || getPadding().empty()) return emitOpError("requires positive groups and non-empty kernel/stride/padding");
  if (!ShapedType::isDynamic(in.getDimSize(1)) && in.getDimSize(1) % getGroups() != 0) return emitOpError("requires input channels divisible by groups");
  if (!ShapedType::isDynamic(out.getDimSize(1)) && out.getDimSize(1) % getGroups() != 0) return emitOpError("requires output channels divisible by groups");
  return success();
}
LogicalResult LinearOp::verify() {
  auto in = tensorType(getInput()), out = tensorType(getOutput());
  if (!in || !out || in.getRank() < 1 || out.getRank() < 1) return emitOpError("requires ranked tensors");
  if (getInFeatures() <= 0 || getOutFeatures() <= 0) return emitOpError("requires positive feature counts");
  if (in.getRank() == out.getRank() && !ShapedType::isDynamic(in.getDimSize(in.getRank()-1)) && in.getDimSize(in.getRank()-1) != getInFeatures()) return emitOpError("input final dimension must equal in_features when rank is unchanged");
  // 保持 frontend 的显式 head 维；仅在 rank 未改变时检查末维，rank 改变时
  // 由头数拆分/合并表达同一个 out_features，不能误判为不合法。
  if (out.getRank() == in.getRank() && !compatible(out.getDimSize(out.getRank()-1), static_cast<int64_t>(getOutFeatures())))
    return emitOpError("output final dimension must equal out_features when rank is unchanged");
  return success();
}
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
LogicalResult ResidualOp::verify() { if (failed(sameShape(*this, getLhs(), getRhs())) || failed(sameShape(*this, getLhs(), getOutput()))) return failure(); return success(); }
LogicalResult PoolOp::verify() { if (getKind() != "avg" && getKind() != "max") return emitOpError("kind must be avg or max"); return success(); }
LogicalResult RescaleOp::verify() { if (getScale().convertToDouble() == 0.0) return emitOpError("scale must be non-zero"); return success(); }
LogicalResult LIFOp::verify() {
  if (getThreshold().convertToDouble() <= 0.0 || getTau().convertToDouble() <= 0.0) return emitOpError("requires positive threshold and tau");
  if (!encodingIs(getOutput().getType(), snn::SpikeEncoding::Binary)) return emitOpError("binary LIF output must use !snn.spike<binary>");
  return success();
}
LogicalResult STBIFOp::verify() {
  if (getThreshold().convertToDouble() <= 0.0 || getTrMin().compare(getTrMax()) == llvm::APFloat::cmpGreaterThan) return emitOpError("requires positive threshold and tr_min <= tr_max");
  if (!encodingIs(getOutput().getType(), snn::SpikeEncoding::Ternary)) return emitOpError("ST-BIF output must use !snn.spike<ternary>");
  return success();
}

#define GET_OP_CLASSES
#include "SNNOp/SNNOpOps.cpp.inc"

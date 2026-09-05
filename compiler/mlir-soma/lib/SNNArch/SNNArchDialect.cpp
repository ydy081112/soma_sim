#include "SNNArch/SNNArchDialect.h"
#include "SNNArch/SNNArchOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace snn_arch;

#include "SNNArch/SNNArchDialect.cpp.inc"

void SNNArchDialect::initialize() {
  addOperations<CoreTypeOp, Conv2DCoreOp, LinearCoreOp, QCoreOp, KCoreOp,
                VCoreOp, ZCoreOp, FCCoreOp, AffineCoreOp, NormCoreOp, QKCoreOp,
                QKVCoreOp, ResidualCoreOp, RescaleCoreOp, PoolCoreOp>();
}

ParseResult CoreTypeOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symbol;
  if (parser.parseSymbolName(symbol, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseLBrace() || parser.parseKeyword("neuron_capacity") ||
      parser.parseEqual())
    return failure();
  int64_t capacity = 0;
  if (parser.parseInteger(capacity)) return failure();
  result.addAttribute("neuron_capacity",
                      IntegerAttr::get(IntegerType::get(parser.getContext(), 64),
                                       capacity));
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("neuron_model") || parser.parseEqual())
      return failure();
    std::string neuronModel;
    if (parser.parseString(&neuronModel)) return failure();
    result.addAttribute("neuron_model",
                        StringAttr::get(parser.getContext(), neuronModel));
  }
  if (parser.parseRBrace()) return failure();
  return success();
}

void CoreTypeOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << " {neuron_capacity = " << getNeuronCapacity();
  if (auto model = (*this)->getAttrOfType<StringAttr>("neuron_model"))
    printer << ", neuron_model = \"" << model.getValue() << '\"';
  printer << '}';
}

LogicalResult CoreTypeOp::verify() {
  if (getNeuronCapacity() <= 0)
    return emitOpError("neuron_capacity must be positive");
  return success();
}

namespace {

static LogicalResult verifyModelCore(Operation *op, unsigned logicalOperands) {
  auto coreType = op->getAttrOfType<FlatSymbolRefAttr>("core_type");
  auto coreID = op->getAttrOfType<IntegerAttr>("core_id");
  auto coord = op->getAttrOfType<DenseI64ArrayAttr>("coord");
  auto partitionID = op->getAttrOfType<IntegerAttr>("partition_id");
  auto offset = op->getAttrOfType<IntegerAttr>("partition_offset");
  auto size = op->getAttrOfType<IntegerAttr>("partition_size");
  if (!coreType || !coreID || !coord || !partitionID || !offset || !size)
    return op->emitOpError("requires all core placement attributes");
  auto declaration = SymbolTable::lookupNearestSymbolFrom<CoreTypeOp>(op, coreType);
  if (!declaration)
    return op->emitOpError("core_type must resolve to snn_arch.core_type");
  if (coreID.getInt() < 0 || partitionID.getInt() < 0 ||
      offset.getInt() < 0 || size.getInt() <= 0)
    return op->emitOpError("core_id/partition_id/offset must be non-negative and size positive");
  if (coord.size() != 2 || coord[0] < 0 || coord[1] < 0)
    return op->emitOpError("coord must contain exactly two non-negative values");
  auto sourceOperand = op->getAttrOfType<DenseI64ArrayAttr>("source_operand");
  auto sourcePartition = op->getAttrOfType<DenseI64ArrayAttr>("source_partition");
  auto sourceOffset = op->getAttrOfType<DenseI64ArrayAttr>("source_offset");
  auto sourceSize = op->getAttrOfType<DenseI64ArrayAttr>("source_size");
  if (!sourceOperand || !sourcePartition || !sourceOffset || !sourceSize ||
      sourceOperand.size() != op->getNumOperands() ||
      sourcePartition.size() != op->getNumOperands() ||
      sourceOffset.size() != op->getNumOperands() ||
      sourceSize.size() != op->getNumOperands())
    return op->emitOpError("source metadata arrays must match the variadic operand count");
  SmallVector<bool> seen(logicalOperands, false);
  for (unsigned index = 0; index < op->getNumOperands(); ++index) {
    int64_t role = sourceOperand[index];
    if (role < 0 || role >= logicalOperands)
      return op->emitOpError("source_operand is outside the logical operand range");
    if (sourcePartition[index] < -1 || sourceOffset[index] < 0 ||
        sourceSize[index] <= 0)
      return op->emitOpError("source partition metadata is invalid");
    seen[role] = true;
  }
  if (llvm::is_contained(seen, false))
    return op->emitOpError("each logical operand must have at least one source partition");
  if (op->getNumResults() < 1 || op->getNumResults() > 2)
    return op->emitOpError("expects one spike result and optional tracer result");
  return success();
}

} // namespace

#define VERIFY_MODEL_CORE(OP, ARITY)                                           \
  LogicalResult OP::verify() { return verifyModelCore(getOperation(), ARITY); }
VERIFY_MODEL_CORE(Conv2DCoreOp, 1)
VERIFY_MODEL_CORE(LinearCoreOp, 1)
VERIFY_MODEL_CORE(QCoreOp, 1)
VERIFY_MODEL_CORE(KCoreOp, 1)
VERIFY_MODEL_CORE(VCoreOp, 1)
VERIFY_MODEL_CORE(ZCoreOp, 1)
VERIFY_MODEL_CORE(FCCoreOp, 1)
VERIFY_MODEL_CORE(AffineCoreOp, 1)
VERIFY_MODEL_CORE(NormCoreOp, 1)
VERIFY_MODEL_CORE(QKCoreOp, 4)
VERIFY_MODEL_CORE(QKVCoreOp, 4)
VERIFY_MODEL_CORE(ResidualCoreOp, 2)
VERIFY_MODEL_CORE(RescaleCoreOp, 1)
VERIFY_MODEL_CORE(PoolCoreOp, 1)
#undef VERIFY_MODEL_CORE

static void nameCoreResults(Operation *op, OpAsmSetValueNameFn setNameFn) {
  for (Value result : op->getResults()) setNameFn(result, "");
}

#define DEFINE_CORE_RESULT_NAMES(OP)                                          \
  void OP::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {                 \
    nameCoreResults(getOperation(), setNameFn);                               \
  }
DEFINE_CORE_RESULT_NAMES(Conv2DCoreOp)
DEFINE_CORE_RESULT_NAMES(LinearCoreOp)
DEFINE_CORE_RESULT_NAMES(QCoreOp)
DEFINE_CORE_RESULT_NAMES(KCoreOp)
DEFINE_CORE_RESULT_NAMES(VCoreOp)
DEFINE_CORE_RESULT_NAMES(ZCoreOp)
DEFINE_CORE_RESULT_NAMES(FCCoreOp)
DEFINE_CORE_RESULT_NAMES(AffineCoreOp)
DEFINE_CORE_RESULT_NAMES(NormCoreOp)
DEFINE_CORE_RESULT_NAMES(QKCoreOp)
DEFINE_CORE_RESULT_NAMES(QKVCoreOp)
DEFINE_CORE_RESULT_NAMES(ResidualCoreOp)
DEFINE_CORE_RESULT_NAMES(RescaleCoreOp)
DEFINE_CORE_RESULT_NAMES(PoolCoreOp)
#undef DEFINE_CORE_RESULT_NAMES

#define GET_OP_CLASSES
#include "SNNArch/SNNArchOps.cpp.inc"

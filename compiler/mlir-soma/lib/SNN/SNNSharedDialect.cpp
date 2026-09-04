#include "SNN/SNNDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace snn;

SpikeType SpikeType::get(MLIRContext *context, SpikeEncoding encoding) {
  return Base::get(context, encoding);
}

VoltageType VoltageType::get(MLIRContext *context, Type elementType) {
  return Base::get(context, elementType);
}

StateType StateType::get(MLIRContext *context, Type payloadType) {
  return Base::get(context, payloadType);
}

SNNDialect::SNNDialect(MLIRContext *context) : Dialect(getDialectNamespace(), context, TypeID::get<SNNDialect>()) {
  addTypes<SpikeType, VoltageType, StateType>();
}

Type SNNDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic))) return {};
  if (failed(parser.parseLess())) return {};
  if (mnemonic == "voltage" || mnemonic == "state") {
    Type wrapped;
    if (failed(parser.parseType(wrapped)) || failed(parser.parseGreater())) return {};
    if (mnemonic == "voltage") {
      if (!isa<IntegerType, FloatType>(wrapped)) {
        parser.emitError(parser.getCurrentLocation(), "voltage requires an integer or floating-point element type");
        return {};
      }
      return VoltageType::get(getContext(), wrapped);
    }
    auto tensor = dyn_cast<RankedTensorType>(wrapped);
    if (!tensor || !isa<VoltageType>(tensor.getElementType())) {
      parser.emitError(parser.getCurrentLocation(), "state requires a ranked tensor of !snn.voltage");
      return {};
    }
    return StateType::get(getContext(), wrapped);
  }
  if (mnemonic != "spike") {
    parser.emitError(parser.getCurrentLocation(), "expected spike, voltage, or state type");
    return {};
  }
  StringRef encoding;
  if (failed(parser.parseKeyword(&encoding)) || failed(parser.parseGreater())) return {};
  if (encoding == "binary") return SpikeType::get(getContext(), SpikeEncoding::Binary);
  if (encoding == "ternary") return SpikeType::get(getContext(), SpikeEncoding::Ternary);
  parser.emitError(parser.getCurrentLocation(), "expected binary or ternary spike encoding");
  return {};
}

void SNNDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto spike = dyn_cast<SpikeType>(type)) {
    printer << "spike<" << (spike.getEncoding() == SpikeEncoding::Binary ? "binary" : "ternary") << ">";
    return;
  }
  if (auto voltage = dyn_cast<VoltageType>(type)) {
    printer << "voltage<" << voltage.getElementType() << ">";
    return;
  }
  auto state = cast<StateType>(type);
  printer << "state<" << state.getPayloadType() << ">";
}

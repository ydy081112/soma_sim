#include "SNN/SNNDialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace snn;

SpikeType SpikeType::get(MLIRContext *context, SpikeEncoding encoding) {
  return Base::get(context, encoding);
}

SNNDialect::SNNDialect(MLIRContext *context) : Dialect(getDialectNamespace(), context, TypeID::get<SNNDialect>()) {
  addTypes<SpikeType>();
}

Type SNNDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)) || mnemonic != "spike") return {};
  if (failed(parser.parseLess())) return {};
  StringRef encoding;
  if (failed(parser.parseKeyword(&encoding)) || failed(parser.parseGreater())) return {};
  if (encoding == "binary") return SpikeType::get(getContext(), SpikeEncoding::Binary);
  if (encoding == "ternary") return SpikeType::get(getContext(), SpikeEncoding::Ternary);
  parser.emitError(parser.getCurrentLocation(), "expected binary or ternary spike encoding");
  return {};
}

void SNNDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto spike = cast<SpikeType>(type);
  printer << "spike<" << (spike.getEncoding() == SpikeEncoding::Binary ? "binary" : "ternary") << ">";
}

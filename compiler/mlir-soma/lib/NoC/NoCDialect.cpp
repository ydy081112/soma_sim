#include "NoC/NoCDialect.h"
#include "NoC/NoCOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include <cmath>
#include <iomanip>
#include <regex>
#include <sstream>

using namespace mlir;
using namespace noc;

#include "NoC/NoCDialect.cpp.inc"

TimeAttr TimeAttr::get(MLIRContext *context, StringRef spelling) {
  return Base::get(context, StringAttr::get(context, spelling));
}

void NoCDialect::initialize() {
  addAttributes<TimeAttr>();
  addOperations<NetworkOp, SendRouterOp, RecvRouterOp>();
}

Attribute NoCDialect::parseAttribute(DialectAsmParser &parser, Type) const {
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)) || mnemonic != "time" ||
      failed(parser.parseLess()))
    return {};
  std::string spelling;
  if (failed(parser.parseString(&spelling)) || failed(parser.parseGreater()))
    return {};
  return TimeAttr::get(getContext(), spelling);
}

void NoCDialect::printAttribute(Attribute attribute,
                                DialectAsmPrinter &printer) const {
  auto time = cast<TimeAttr>(attribute);
  printer << "time<\"" << time.getSpelling() << "\">";
}

namespace {

static bool validDuration(StringRef spelling) {
  static const std::regex pattern(R"(^((0|[1-9][0-9]*)(\.[0-9]+)?)(ps|ns|us|ms|s)$)");
  return std::regex_match(spelling.str(), pattern);
}

static std::string formatDuration(double value, StringRef unit) {
  std::ostringstream stream;
  stream << std::setprecision(15) << value;
  return stream.str() + unit.str();
}

static ParseResult parseDuration(OpAsmParser &parser, TimeAttr &result) {
  double value = 0.0;
  int64_t integerValue = 0;
  OptionalParseResult integer = parser.parseOptionalInteger(integerValue);
  if (!integer.has_value()) {
    if (failed(parser.parseFloat(value))) return failure();
  } else if (failed(*integer)) {
    return failure();
  } else {
    value = static_cast<double>(integerValue);
  }
  StringRef unit;
  if (failed(parser.parseKeyword(&unit))) return failure();
  if (!std::isfinite(value) || value < 0.0 ||
      (unit != "ps" && unit != "ns" && unit != "us" && unit != "ms" &&
       unit != "s"))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected a non-negative duration with ps/ns/us/ms/s unit");
  result = TimeAttr::get(parser.getContext(), formatDuration(value, unit));
  return success();
}

} // namespace

ParseResult NetworkOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr symbol;
  if (parser.parseSymbolName(symbol, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseLBrace() || parser.parseKeyword("topology") ||
      parser.parseEqual())
    return failure();
  std::string topology;
  if (parser.parseString(&topology) || parser.parseComma() ||
      parser.parseKeyword("dimensions") || parser.parseEqual() ||
      parser.parseLSquare())
    return failure();
  int64_t rows = 0, columns = 0;
  if (parser.parseInteger(rows) || parser.parseComma() ||
      parser.parseInteger(columns) || parser.parseRSquare() ||
      parser.parseComma() || parser.parseKeyword("hop_latency") ||
      parser.parseEqual())
    return failure();
  TimeAttr hopLatency;
  if (failed(parseDuration(parser, hopLatency)) || parser.parseComma() ||
      parser.parseKeyword("routing") || parser.parseEqual())
    return failure();
  std::string routing;
  if (parser.parseString(&routing) || parser.parseRBrace()) return failure();

  MLIRContext *context = parser.getContext();
  result.addAttribute("topology", StringAttr::get(context, topology));
  result.addAttribute("dimensions", DenseI64ArrayAttr::get(context, {rows, columns}));
  result.addAttribute("hop_latency", hopLatency);
  result.addAttribute("routing", StringAttr::get(context, routing));
  return success();
}

void NetworkOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printSymbolName(getSymName());
  printer << " {topology = \"" << getTopology() << "\", dimensions = [";
  auto dimensions = getDimensions();
  printer << dimensions[0] << ", " << dimensions[1] << "], hop_latency = ";
  printer << cast<TimeAttr>(getHopLatency()).getSpelling();
  printer << ", routing = \"" << getRouting() << "\"}";
}

LogicalResult NetworkOp::verify() {
  if (getTopology() != "mesh")
    return emitOpError("core-level NoC only supports topology=\"mesh\"");
  if (getRouting() != "xy")
    return emitOpError("core-level NoC only supports routing=\"xy\"");
  auto dimensions = getDimensions();
  if (dimensions.size() != 2 || dimensions[0] <= 0 || dimensions[1] <= 0)
    return emitOpError("dimensions must contain exactly two positive values");
  auto time = dyn_cast<TimeAttr>(getHopLatency());
  if (!time || !validDuration(time.getSpelling()))
    return emitOpError("hop_latency must be a non-negative ps/ns/us/ms/s duration");
  return success();
}

namespace {

static LogicalResult verifyRouter(Operation *op) {
  if (op->getOperand(0).getType() != op->getResult(0).getType())
    return op->emitOpError("must preserve the routed SSA type");
  auto network = op->getAttrOfType<FlatSymbolRefAttr>("network");
  auto declaration = SymbolTable::lookupNearestSymbolFrom<NetworkOp>(op, network);
  if (!declaration)
    return op->emitOpError("network must resolve to noc.network");
  auto coord = op->getAttrOfType<DenseI64ArrayAttr>("coord");
  auto dimensions = declaration.getDimensions();
  if (!coord || coord.size() != 2 || coord[0] < 0 || coord[1] < 0 ||
      coord[0] >= dimensions[0] || coord[1] >= dimensions[1])
    return op->emitOpError("coord is outside the referenced network dimensions");
  return success();
}

} // namespace

LogicalResult SendRouterOp::verify() { return verifyRouter(getOperation()); }
LogicalResult RecvRouterOp::verify() { return verifyRouter(getOperation()); }

#define GET_OP_CLASSES
#include "NoC/NoCOps.cpp.inc"

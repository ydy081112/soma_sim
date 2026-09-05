#include "SNNArch/SNNArchPasses.h"

#include "NoC/NoCOps.h"
#include "SNNArch/SNNArchOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include <limits>

using namespace mlir;

namespace snn_arch {
namespace {

struct Placement {
  int64_t coreID;
  int64_t partitionID;
  int64_t offset;
  int64_t size;
};

struct PartValue {
  Value value;
  int64_t coreID;
  int64_t partitionID;
  int64_t offset;
  int64_t size;
};

static std::optional<StringRef> getCoreMnemonic(Operation *op) {
  static const llvm::StringMap<StringRef> mapping = {
      {"snn_op.conv2d_stbif", "snn_arch.conv2d_core"},
      {"snn_op.linear_stbif", "snn_arch.linear_core"},
      {"snn_op.q_stbif", "snn_arch.q_core"},
      {"snn_op.k_stbif", "snn_arch.k_core"},
      {"snn_op.v_stbif", "snn_arch.v_core"},
      {"snn_op.z_stbif", "snn_arch.z_core"},
      {"snn_op.fc_stbif", "snn_arch.fc_core"},
      {"snn_op.affine_stbif", "snn_arch.affine_core"},
      {"snn_op.norm_stbif", "snn_arch.norm_core"},
      {"snn_op.qk_stbif", "snn_arch.qk_core"},
      {"snn_op.qkv_stbif", "snn_arch.qkv_core"},
      {"snn_op.residual_stbif", "snn_arch.residual_core"},
      {"snn_op.rescale_stbif", "snn_arch.rescale_core"},
      {"snn_op.pool_stbif", "snn_arch.pool_core"},
  };
  auto found = mapping.find(op->getName().getStringRef());
  if (found == mapping.end()) return std::nullopt;
  return found->second;
}

static FailureOr<int64_t> getPopulationSize(Operation *op) {
  if (op->getNumResults() == 0)
    return op->emitOpError("cannot map an operation without a neuron result");
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!type || !type.hasStaticShape())
    return op->emitOpError("core mapping requires a statically shaped tensor result");
  auto timeDimAttr = op->getAttrOfType<IntegerAttr>("time_dim");
  if (!timeDimAttr)
    return op->emitOpError("core mapping requires time_dim");
  int64_t timeDim = timeDimAttr.getInt();
  if (timeDim < 0) timeDim += type.getRank();
  if (timeDim < 0 || timeDim >= type.getRank())
    return op->emitOpError("time_dim is outside the result rank");
  int64_t population = 1;
  for (auto [index, dimension] : llvm::enumerate(type.getShape())) {
    if (static_cast<int64_t>(index) == timeDim) continue;
    if (dimension <= 0 || population > std::numeric_limits<int64_t>::max() / dimension)
      return op->emitOpError("invalid or overflowing neuron population shape");
    population *= dimension;
  }
  return population;
}

static int64_t getValuePopulation(Value value, int64_t timeDim) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape()) return 1;
  if (timeDim < 0) timeDim += type.getRank();
  int64_t population = 1;
  for (auto [index, dimension] : llvm::enumerate(type.getShape()))
    if (static_cast<int64_t>(index) != timeDim) population *= dimension;
  return population;
}

static bool usesAlignedPartitions(StringRef coreMnemonic) {
  return coreMnemonic == "snn_arch.affine_core" ||
         coreMnemonic == "snn_arch.norm_core" ||
         coreMnemonic == "snn_arch.rescale_core" ||
         coreMnemonic == "snn_arch.residual_core";
}

static Operation *createOperation(OpBuilder &builder, Location loc,
                                  StringRef name, ValueRange operands,
                                  TypeRange results, ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attrs);
  return builder.create(state);
}

struct SNNOpCoreMappingPass final
    : PassWrapper<SNNOpCoreMappingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SNNOpCoreMappingPass)

  SNNOpCoreMappingPass() = default;
  SNNOpCoreMappingPass(const SNNOpCoreMappingPass &other)
      : PassWrapper(other) {}

  Option<std::string> hardwareFile{
      *this, "hardware-file",
      llvm::cl::desc("core-level hardware MLIR containing noc.network and snn_arch.core_type"),
      llvm::cl::init("")};
  Option<std::string> networkSymbol{
      *this, "network", llvm::cl::desc("noc.network symbol name"),
      llvm::cl::init("noc0")};
  Option<std::string> coreTypeSymbol{
      *this, "core-type", llvm::cl::desc("snn_arch.core_type symbol name"),
      llvm::cl::init("standard_core")};

  StringRef getArgument() const final { return "snnop-core-mapping"; }
  StringRef getDescription() const final {
    return "partition fused SNNOp populations onto model-aware core operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<noc::NoCDialect, SNNArchDialect>();
  }

  LogicalResult importHardware(ModuleOp module) {
    if (hardwareFile.empty()) return success();
    ParserConfig config(&getContext());
    OwningOpRef<ModuleOp> source =
        parseSourceFile<ModuleOp>(StringRef(hardwareFile), config);
    if (!source)
      return module.emitError() << "failed to parse hardware file " << hardwareFile;
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(module.getBody());
    for (Operation &op : source->getBody()->without_terminator()) {
      auto symbol = op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
      if (!symbol || (!isa<noc::NetworkOp>(op) && !isa<CoreTypeOp>(op))) continue;
      if (SymbolTable::lookupSymbolIn(module, symbol.getValue())) {
        op.emitError() << "hardware symbol @" << symbol.getValue()
                       << " already exists in the input module";
        return failure();
      }
      builder.insert(op.clone());
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(importHardware(module))) return signalPassFailure();

    auto coreType = module.lookupSymbol<CoreTypeOp>(coreTypeSymbol);
    auto network = module.lookupSymbol<noc::NetworkOp>(networkSymbol);
    if (!coreType || !network) {
      module.emitError() << "requires snn_arch.core_type @" << coreTypeSymbol
                         << " and noc.network @" << networkSymbol;
      return signalPassFailure();
    }
    coreType->setAttr("neuron_model", StringAttr::get(&getContext(), "st_bif"));
    int64_t capacity = coreType.getNeuronCapacity();
    auto dimensions = network.getDimensions();
    int64_t meshCapacity = dimensions[0] * dimensions[1];

    SmallVector<Operation *> fusedOps;
    Operation *unsupportedFused = nullptr;
    module.walk([&](Operation *op) {
      StringRef name = op->getName().getStringRef();
      if (!name.starts_with("snn_op.") || !name.ends_with("_stbif")) return;
      if (getCoreMnemonic(op))
        fusedOps.push_back(op);
      else if (!unsupportedFused)
        unsupportedFused = op;
    });
    if (unsupportedFused) {
      unsupportedFused->emitOpError("has no Core-level mapping");
      return signalPassFailure();
    }
    if (fusedOps.empty()) {
      module.emitError("contains no supported fused SNNOp operations");
      return signalPassFailure();
    }

    DenseMap<Operation *, SmallVector<Placement>> plans;
    SmallVector<int64_t> usedCapacity;
    for (Operation *op : fusedOps) {
      FailureOr<int64_t> population = getPopulationSize(op);
      if (failed(population)) return signalPassFailure();
      int64_t offset = 0;
      int64_t partitionID = 0;
      while (offset < *population) {
        int64_t size = std::min(capacity, *population - offset);
        int64_t coreID = -1;
        for (auto [candidate, used] : llvm::enumerate(usedCapacity)) {
          if (capacity - used >= size) {
            coreID = candidate;
            break;
          }
        }
        if (coreID < 0) {
          coreID = usedCapacity.size();
          if (coreID >= meshCapacity) {
            op->emitOpError() << "mapping requires more than " << meshCapacity
                              << " NoC mesh cores";
            return signalPassFailure();
          }
          usedCapacity.push_back(0);
        }
        usedCapacity[coreID] += size;
        plans[op].push_back({coreID, partitionID++, offset, size});
        offset += size;
      }
    }

    DenseMap<Value, SmallVector<PartValue>> mappedResults;
    OpBuilder builder(&getContext());
    auto coreRef = FlatSymbolRefAttr::get(&getContext(), coreTypeSymbol);
    auto networkRef = FlatSymbolRefAttr::get(&getContext(), networkSymbol);

    auto coordFor = [&](int64_t coreID) {
      return SmallVector<int64_t>{coreID % dimensions[0], coreID / dimensions[0]};
    };

    auto route = [&](PartValue source, int64_t destinationCore,
                     Location loc) -> Value {
      if (source.coreID == destinationCore) return source.value;
      SmallVector<NamedAttribute> sendAttrs{
          builder.getNamedAttr("network", networkRef),
          builder.getNamedAttr("coord", builder.getDenseI64ArrayAttr(coordFor(source.coreID)))};
      Operation *send = createOperation(builder, loc, "noc.send_router",
                                        source.value, source.value.getType(), sendAttrs);
      SmallVector<NamedAttribute> recvAttrs{
          builder.getNamedAttr("network", networkRef),
          builder.getNamedAttr("coord", builder.getDenseI64ArrayAttr(coordFor(destinationCore)))};
      return createOperation(builder, loc, "noc.recv_router", send->getResult(0),
                             source.value.getType(), recvAttrs)->getResult(0);
    };

    for (Operation *op : fusedOps) {
      builder.setInsertionPoint(op);
      StringRef coreMnemonic = *getCoreMnemonic(op);
      for (const Placement &placement : plans[op]) {
        SmallVector<Value> operands;
        SmallVector<int64_t> sourceOperands, sourcePartitions, sourceOffsets,
            sourceSizes;
        int64_t timeDim = op->getAttrOfType<IntegerAttr>("time_dim").getInt();
        for (auto [operandIndex, operand] : llvm::enumerate(op->getOperands())) {
          auto found = mappedResults.find(operand);
          if (found == mappedResults.end()) {
            operands.push_back(operand);
            sourceOperands.push_back(operandIndex);
            sourcePartitions.push_back(-1);
            sourceOffsets.push_back(0);
            sourceSizes.push_back(getValuePopulation(operand, timeDim));
            continue;
          }
          bool aligned = usesAlignedPartitions(coreMnemonic);
          SmallVector<const PartValue *> selected;
          for (const PartValue &part : found->second) {
            bool overlaps = part.offset < placement.offset + placement.size &&
                            placement.offset < part.offset + part.size;
            if (!aligned || overlaps) selected.push_back(&part);
          }
          // Shape-changing model ops conservatively consume every source
          // partition. Aligned elementwise ops consume only overlapping slices.
          if (selected.empty())
            for (const PartValue &part : found->second) selected.push_back(&part);
          for (const PartValue *part : selected) {
            operands.push_back(route(*part, placement.coreID, op->getLoc()));
            sourceOperands.push_back(operandIndex);
            sourcePartitions.push_back(part->partitionID);
            sourceOffsets.push_back(part->offset);
            sourceSizes.push_back(part->size);
          }
        }

        SmallVector<NamedAttribute> attrs(op->getAttrs().begin(), op->getAttrs().end());
        attrs.push_back(builder.getNamedAttr("core_type", coreRef));
        attrs.push_back(builder.getNamedAttr("core_id", builder.getI64IntegerAttr(placement.coreID)));
        attrs.push_back(builder.getNamedAttr("coord", builder.getDenseI64ArrayAttr(coordFor(placement.coreID))));
        attrs.push_back(builder.getNamedAttr("partition_id", builder.getI64IntegerAttr(placement.partitionID)));
        attrs.push_back(builder.getNamedAttr("partition_offset", builder.getI64IntegerAttr(placement.offset)));
        attrs.push_back(builder.getNamedAttr("partition_size", builder.getI64IntegerAttr(placement.size)));
        attrs.push_back(builder.getNamedAttr("source_operand", builder.getDenseI64ArrayAttr(sourceOperands)));
        attrs.push_back(builder.getNamedAttr("source_partition", builder.getDenseI64ArrayAttr(sourcePartitions)));
        attrs.push_back(builder.getNamedAttr("source_offset", builder.getDenseI64ArrayAttr(sourceOffsets)));
        attrs.push_back(builder.getNamedAttr("source_size", builder.getDenseI64ArrayAttr(sourceSizes)));
        Operation *core = createOperation(builder, op->getLoc(), coreMnemonic,
                                          operands, op->getResultTypes(), attrs);
        for (auto [index, result] : llvm::enumerate(op->getResults()))
          mappedResults[result].push_back(
              {core->getResult(index), placement.coreID, placement.partitionID,
               placement.offset, placement.size});
      }
    }

    // Core-level function boundaries expose partitioned results directly.  No
    // implicit join hides the partition-to-partition SSA graph.
    llvm::SmallDenseSet<Operation *> fusedSet(fusedOps.begin(), fusedOps.end());
    module.walk([&](func::ReturnOp returnOp) {
      SmallVector<Value> partitionedReturns;
      for (Value operand : returnOp.getOperands()) {
        auto found = mappedResults.find(operand);
        if (found == mappedResults.end()) {
          partitionedReturns.push_back(operand);
          continue;
        }
        for (const PartValue &part : found->second)
          partitionedReturns.push_back(part.value);
      }
      returnOp->setOperands(partitionedReturns);
      auto function = returnOp->getParentOfType<func::FuncOp>();
      SmallVector<Type> resultTypes;
      for (Value value : partitionedReturns) resultTypes.push_back(value.getType());
      function.setFunctionType(FunctionType::get(
          &getContext(), function.getFunctionType().getInputs(), resultTypes));
    });
    for (Operation *op : fusedOps)
      for (Value result : op->getResults())
        for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
          if (!fusedSet.contains(use.getOwner())) {
            use.getOwner()->emitError("unsupported non-return use of a mapped fused result");
            return signalPassFailure();
          }
          use.set(mappedResults[result].front().value);
        }
    for (Operation *op : llvm::reverse(fusedOps)) op->erase();
  }
};

} // namespace

void registerSNNOpCoreMappingPass() {
  PassRegistration<SNNOpCoreMappingPass>();
}

void registerSNNArchPasses() { registerSNNOpCoreMappingPass(); }

} // namespace snn_arch

#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace snn {

enum class SpikeEncoding : unsigned char { Binary, Ternary };

class SpikeTypeStorage : public mlir::TypeStorage {
public:
  using KeyTy = SpikeEncoding;
  explicit SpikeTypeStorage(SpikeEncoding encoding) : encoding(encoding) {}
  bool operator==(const KeyTy &key) const { return key == encoding; }
  static SpikeTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                     const KeyTy &key) {
    return new (allocator.allocate<SpikeTypeStorage>()) SpikeTypeStorage(key);
  }
  SpikeEncoding encoding;
};

class SpikeType : public mlir::Type::TypeBase<SpikeType, mlir::Type,
                                               SpikeTypeStorage> {
public:
  static constexpr llvm::StringLiteral name = "snn.spike";
  using Base::Base;
  static SpikeType get(mlir::MLIRContext *context, SpikeEncoding encoding);
  SpikeEncoding getEncoding() const { return getImpl()->encoding; }
};

class SNNDialect : public mlir::Dialect {
public:
  explicit SNNDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "snn"; }
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;
};
} // namespace snn

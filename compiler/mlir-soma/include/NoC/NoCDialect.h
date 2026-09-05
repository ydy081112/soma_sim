#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

namespace noc {

class TimeAttrStorage : public mlir::AttributeStorage {
public:
  using KeyTy = mlir::StringAttr;
  explicit TimeAttrStorage(mlir::StringAttr spelling) : spelling(spelling) {}
  bool operator==(const KeyTy &key) const { return key == spelling; }
  static TimeAttrStorage *construct(mlir::AttributeStorageAllocator &allocator,
                                    const KeyTy &key) {
    return new (allocator.allocate<TimeAttrStorage>()) TimeAttrStorage(key);
  }
  mlir::StringAttr spelling;
};

/// 带单位的时间字面量。第一层保持前端文本；后续硬件层可再转换为 ps。
class TimeAttr : public mlir::Attribute::AttrBase<TimeAttr, mlir::Attribute,
                                                  TimeAttrStorage> {
public:
  static constexpr llvm::StringLiteral name = "noc.time";
  using Base::Base;
  static TimeAttr get(mlir::MLIRContext *context, llvm::StringRef spelling);
  llvm::StringRef getSpelling() const { return getImpl()->spelling.getValue(); }
};

} // namespace noc

#include "NoC/NoCDialect.h.inc"

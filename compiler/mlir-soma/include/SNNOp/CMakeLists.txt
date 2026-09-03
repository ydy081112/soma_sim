set(LLVM_TARGET_DEFINITIONS SNNOpDialect.td)
mlir_tablegen(SNNOpDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SNNOpDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS SNNOpOps.td)
mlir_tablegen(SNNOpOps.h.inc -gen-op-decls)
mlir_tablegen(SNNOpOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(SNNOpIncGen)

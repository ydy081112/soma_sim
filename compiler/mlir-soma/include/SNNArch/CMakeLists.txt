set(LLVM_TARGET_DEFINITIONS SNNArchDialect.td)
mlir_tablegen(SNNArchDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SNNArchDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS SNNArchOps.td)
mlir_tablegen(SNNArchOps.h.inc -gen-op-decls)
mlir_tablegen(SNNArchOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(SNNArchIncGen)

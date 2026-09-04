set(LLVM_TARGET_DEFINITIONS SNNExecDialect.td)
mlir_tablegen(SNNExecDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SNNExecDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS SNNExecOps.td)
mlir_tablegen(SNNExecOps.h.inc -gen-op-decls)
mlir_tablegen(SNNExecOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(SNNExecIncGen)

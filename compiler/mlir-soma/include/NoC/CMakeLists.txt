set(LLVM_TARGET_DEFINITIONS NoCDialect.td)
mlir_tablegen(NoCDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NoCDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS NoCOps.td)
mlir_tablegen(NoCOps.h.inc -gen-op-decls)
mlir_tablegen(NoCOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(NoCIncGen)

set(LLVM_TARGET_DEFINITIONS SNNTypes.td)
mlir_tablegen(SNNTypes.h.inc -gen-typedef-decls -typedefs-dialect=snn)
mlir_tablegen(SNNTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=snn)
add_public_tablegen_target(SNNTypesIncGen)

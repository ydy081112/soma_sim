add_mlir_tool(soma-opt soma-opt.cpp)
target_link_libraries(soma-opt PRIVATE SNNSharedDialect SNNOpDialect SNNOpTransforms
  SNNExecDialect SNNExecTransforms NoCDialect SNNArchDialect SNNArchTransforms MLIRFuncDialect MLIRTensorDialect
  MLIRArithDialect MLIRParser MLIRPass MLIRIR MLIRSupport MLIRMlirOptMain MLIROptLib)

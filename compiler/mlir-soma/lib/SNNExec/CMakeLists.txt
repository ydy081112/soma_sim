add_mlir_dialect_library(SNNExecDialect SNNExecDialect.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS SNNExecIncGen
  LINK_LIBS PUBLIC MLIRIR SNNSharedDialect SNNOpDialect)

add_mlir_library(SNNExecTransforms
  LowerSNNOpToSNNExec.cpp
  LINK_LIBS PUBLIC SNNExecDialect SNNOpDialect SNNSharedDialect
                   MLIRFuncDialect MLIRTransformUtils MLIRPass)

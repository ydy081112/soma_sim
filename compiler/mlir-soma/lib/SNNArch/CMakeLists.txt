add_mlir_dialect_library(SNNArchDialect SNNArchDialect.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS SNNArchIncGen
  LINK_LIBS PUBLIC MLIRIR)

add_mlir_library(SNNArchTransforms
  SNNOpCoreMapping.cpp
  LINK_LIBS PUBLIC SNNArchDialect NoCDialect SNNOpDialect
    MLIRFuncDialect MLIRParser MLIRPass MLIRSupport)

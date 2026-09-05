add_mlir_dialect_library(NoCDialect NoCDialect.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS NoCIncGen
  LINK_LIBS PUBLIC MLIRIR MLIRFuncDialect)

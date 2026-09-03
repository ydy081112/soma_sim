add_mlir_dialect_library(SNNSharedDialect SNNSharedDialect.cpp
  DEPENDS SNNTypesIncGen
  LINK_LIBS PUBLIC MLIRIR)

add_mlir_dialect_library(SNNOpDialect SNNOpDialect.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS SNNOpIncGen
  LINK_LIBS PUBLIC MLIRIR SNNSharedDialect)
add_mlir_library(SNNOpTransforms SNNOpCanonicalize.cpp
  LINK_LIBS PUBLIC SNNOpDialect MLIRTransformUtils MLIRPass)

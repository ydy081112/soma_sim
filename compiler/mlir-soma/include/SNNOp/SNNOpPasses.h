#pragma once
#include "mlir/Pass/Pass.h"
namespace snn_op {
void registerSNNOpPasses();
void registerModelNeuronFusionPass();
void registerDeadNeuronOutEliminatePass();
} // namespace snn_op

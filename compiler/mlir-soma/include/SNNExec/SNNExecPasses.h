#pragma once

namespace snn_exec {

void registerLowerSNNOpToSNNExecPass();

inline void registerSNNExecPasses() { registerLowerSNNOpToSNNExecPass(); }

} // namespace snn_exec

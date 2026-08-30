#pragma once

#include "soma/runtime/weight_store.hpp"

#include <cstdint>
#include <stdexcept>

namespace soma {

class SynapseEngine {
public:
    // apply 统一 Linear 和 Spatial 两条热路径，并通过回调写入 Soma SoA。
    template <typename Fn>
    static std::uint64_t apply(const LayerWeights& weights, std::uint64_t source_neuron,
                               float value, std::uint64_t destination_neurons, Fn&& update) {
        if (weights.op == LayerOp::Linear) {
            // source-major [Cin,Cout] 让一个 source spike 顺序读取一整行权重。
            const auto source_neurons = weights.dense_weight.size() / destination_neurons;
            if (source_neuron >= source_neurons) throw std::runtime_error("dense source neuron 越界");
            const auto base = source_neuron * destination_neurons;
            for (std::uint64_t destination = 0; destination < destination_neurons; ++destination) {
                update(destination, value * weights.dense_weight[base + destination]);
            }
            return destination_neurons;
        }
        return weights.spatial.for_each_destination(source_neuron, value, std::forward<Fn>(update));
    }
};

}  // namespace soma

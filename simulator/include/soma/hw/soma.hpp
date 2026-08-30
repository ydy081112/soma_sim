#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace soma {

struct FiredNeuron {
    std::uint64_t neuron = 0;
    float value = 1.0F;
};

struct SomaNeuronResult {
    bool updated = false;
    std::optional<FiredNeuron> fired;
};

// Voltage/threshold 分离为连续 SoA；没有 per-neuron object。
class SomaState {
public:
    SomaState(std::size_t neurons, float threshold, float leak, std::string reset, bool readout);

    SomaNeuronResult process_neuron(std::uint64_t neuron, float synaptic_input,
                                    bool has_pending_input, float bias);
    const std::vector<float>& voltage() const { return voltage_; }

private:
    std::vector<float> voltage_;
    std::vector<float> threshold_;
    float leak_ = 1.0F;
    std::string reset_;
    bool readout_ = false;
};

}  // namespace soma

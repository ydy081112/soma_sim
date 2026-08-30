#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace soma {

struct FiredNeuron {
    std::uint64_t neuron = 0;
    float value = 1.0F;
};

// Voltage/threshold/last-timestep/active 分离为连续 SoA；没有 per-neuron object。
class SomaState {
public:
    SomaState(std::size_t neurons, float threshold, float leak, std::string reset, bool readout);

    std::uint64_t apply_bias(const std::vector<float>& bias, std::uint32_t output_channels,
                             std::uint32_t timestep);
    void accumulate(std::uint64_t neuron, float delta, std::uint32_t timestep);
    std::optional<FiredNeuron> fire_one();
    bool has_pending() const { return !candidates_.empty(); }
    const std::vector<float>& voltage() const { return voltage_; }

private:
    std::vector<float> voltage_;
    std::vector<float> threshold_;
    std::vector<std::uint32_t> last_timestep_;
    std::vector<std::uint8_t> candidate_active_;
    std::deque<std::uint64_t> candidates_;
    float leak_ = 1.0F;
    std::string reset_;
    bool readout_ = false;
};

}  // namespace soma

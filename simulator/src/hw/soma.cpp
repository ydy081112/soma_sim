#include "soma/hw/soma.hpp"

#include <cmath>
#include <stdexcept>

namespace soma {

SomaState::SomaState(std::size_t neurons, float threshold, float leak, std::string reset, bool readout)
    : voltage_(neurons, 0.0F), threshold_(neurons, threshold), last_timestep_(neurons, 0),
      candidate_active_(neurons, 0), leak_(leak), reset_(std::move(reset)), readout_(readout) {
    if (neurons == 0 || threshold <= 0.0F) throw std::runtime_error("SomaState 参数无效");
    if (reset_ != "soft" && reset_ != "hard") throw std::runtime_error("reset 只支持 soft/hard");
}

std::uint64_t SomaState::apply_bias(const std::vector<float>& bias, std::uint32_t output_channels,
                                    std::uint32_t timestep) {
    if (bias.empty()) return 0;
    if (bias.size() != output_channels || voltage_.size() % output_channels != 0) {
        throw std::runtime_error("bias 与 soma layout 不匹配");
    }
    for (std::size_t neuron = 0; neuron < voltage_.size(); ++neuron) {
        accumulate(neuron, bias[neuron % output_channels], timestep);
    }
    return voltage_.size();
}

void SomaState::accumulate(std::uint64_t neuron, float delta, std::uint32_t timestep) {
    if (neuron >= voltage_.size()) throw std::runtime_error("soma neuron 越界");
    const auto index = static_cast<std::size_t>(neuron);
    if (timestep > last_timestep_[index] && leak_ != 1.0F) {
        voltage_[index] *= std::pow(leak_, static_cast<float>(timestep - last_timestep_[index]));
    }
    last_timestep_[index] = timestep;
    voltage_[index] += delta;
    if (!readout_ && voltage_[index] >= threshold_[index] && !candidate_active_[index]) {
        candidate_active_[index] = 1;
        candidates_.push_back(neuron);
    }
}

std::optional<FiredNeuron> SomaState::fire_one() {
    while (!candidates_.empty()) {
        const auto neuron = candidates_.front();
        candidates_.pop_front();
        const auto index = static_cast<std::size_t>(neuron);
        candidate_active_[index] = 0;
        if (voltage_[index] < threshold_[index]) continue;
        if (reset_ == "soft") voltage_[index] -= threshold_[index];
        else voltage_[index] = 0.0F;
        if (voltage_[index] >= threshold_[index]) {
            candidate_active_[index] = 1;
            candidates_.push_back(neuron);
        }
        return FiredNeuron{neuron, 1.0F};
    }
    return std::nullopt;
}

}  // namespace soma

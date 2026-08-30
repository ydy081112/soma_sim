#include "soma/hw/soma.hpp"

#include <stdexcept>

namespace soma {

SomaState::SomaState(std::size_t neurons, float threshold, float leak, std::string reset, bool readout)
    : voltage_(neurons, 0.0F), threshold_(neurons, threshold),
      leak_(leak), reset_(std::move(reset)), readout_(readout) {
    if (neurons == 0 || threshold <= 0.0F) throw std::runtime_error("SomaState 参数无效");
    if (reset_ != "soft" && reset_ != "hard") throw std::runtime_error("reset 只支持 soft/hard");
}

SomaNeuronResult SomaState::process_neuron(std::uint64_t neuron, float synaptic_input,
                                           bool has_pending_input, float bias) {
    if (neuron >= voltage_.size()) throw std::runtime_error("soma neuron 越界");
    const auto index = static_cast<std::size_t>(neuron);
    // pending 标志保留“收到过但累加和为 0”的更新；已有膜电位也需要继续做 timestep transition。
    SomaNeuronResult result;
    result.updated = has_pending_input || bias != 0.0F || voltage_[index] != 0.0F;
    if (!result.updated) return result;

    voltage_[index] *= leak_;
    voltage_[index] += synaptic_input + bias;
    if (readout_ || voltage_[index] < threshold_[index]) return result;

    // 每个 neuron 每个 timestep 最多 firing 一次；soft reset 的剩余电位留到后续 timestep。
    result.fired = FiredNeuron{neuron, 1.0F};
    if (reset_ == "soft") voltage_[index] -= threshold_[index];
    else voltage_[index] = 0.0F;
    return result;
}

}  // namespace soma

#include "soma/hw/soma.hpp"

#include <algorithm>
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
    // spatial-major state 复用每个 output channel 的 bias，不额外展开 bias 数组。
    for (std::size_t neuron = 0; neuron < voltage_.size(); ++neuron) {
        accumulate(neuron, bias[neuron % output_channels], timestep);
    }
    return voltage_.size();
}

void SomaState::accumulate(std::uint64_t neuron, float delta, std::uint32_t timestep) {
    if (neuron >= voltage_.size()) throw std::runtime_error("soma neuron 越界");
    const auto index = static_cast<std::size_t>(neuron);
    // 只在该 neuron 真正收到更新时补算跨 timestep leak，避免全状态扫描。
    if (timestep > last_timestep_[index] && leak_ != 1.0F) {
        voltage_[index] *= std::pow(leak_, static_cast<float>(timestep - last_timestep_[index]));
    }
    last_timestep_[index] = timestep;
    voltage_[index] += delta;
    // active 位防止同一 Core update 内重复记录同一个候选 neuron。
    if (!readout_ && voltage_[index] >= threshold_[index] && !candidate_active_[index]) {
        candidate_active_[index] = 1;
        candidates_.push_back(neuron);
    }
}

std::vector<FiredNeuron> SomaState::fire_all_ordered() {
    // 只排序本次实际越阈值的候选；不扫描完整 neuron state。
    std::sort(candidates_.begin(), candidates_.end());
    std::vector<FiredNeuron> fired;
    for (const auto neuron : candidates_) {
        const auto index = static_cast<std::size_t>(neuron);
        candidate_active_[index] = 0;
        if (voltage_[index] < threshold_[index]) continue;
        do {
            fired.push_back(FiredNeuron{neuron, 1.0F});
            if (reset_ == "soft") voltage_[index] -= threshold_[index];
            else voltage_[index] = 0.0F;
        } while (reset_ == "soft" && voltage_[index] >= threshold_[index]);
    }
    candidates_.clear();
    return fired;
}

}  // namespace soma

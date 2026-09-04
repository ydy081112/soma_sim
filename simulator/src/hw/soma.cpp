#include "soma/hw/soma.hpp"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace soma {

SomaState::SomaState(std::size_t neurons, float threshold, float leak, std::string reset,
                     bool readout, float membrane_quantization_step,
                     std::string threshold_comparison, std::string threshold_compare_mode,
                     std::vector<float> thresholds,
                     float membrane_min, float membrane_max,
                     bool fire_on_positive_saturation, std::string neuron_model,
                     std::int32_t tracer_min, std::int32_t tracer_max,
                     std::vector<float> initial_membrane)
    : voltage_(initial_membrane.empty() ? std::vector<float>(neurons, 0.0F)
                                        : std::move(initial_membrane)),
      threshold_(thresholds.empty() ? std::vector<float>(neurons, threshold)
                                    : std::move(thresholds)),
      fire_count_(neurons, 0),
      tracer_(neurons, 0),
      leak_(leak), reset_(std::move(reset)), readout_(readout),
      membrane_quantization_step_(membrane_quantization_step),
      threshold_comparison_(std::move(threshold_comparison)),
      threshold_compare_mode_(std::move(threshold_compare_mode)),
      membrane_min_(membrane_min), membrane_max_(membrane_max),
      fire_on_positive_saturation_(fire_on_positive_saturation),
      st_bif_(neuron_model == "st_bif"), tracer_min_(tracer_min), tracer_max_(tracer_max) {
    if (neurons == 0 || threshold <= 0.0F) throw std::runtime_error("SomaState 参数无效");
    if (voltage_.size() != neurons) throw std::runtime_error("SomaState initial membrane shape 无效");
    if (threshold_.size() != neurons ||
        std::any_of(threshold_.begin(), threshold_.end(), [](float value) { return value < 0.0F; }))
        throw std::runtime_error("SomaState threshold array 无效");
    if (reset_ != "soft" && reset_ != "hard") throw std::runtime_error("reset 只支持 soft/hard");
    if (membrane_quantization_step_ < 0.0F) throw std::runtime_error("膜电位量化步长不得为负");
    if (threshold_comparison_ != "greater" && threshold_comparison_ != "greater_equal")
        throw std::runtime_error("threshold comparison 只支持 greater/greater_equal");
    if (threshold_compare_mode_ != "signed" &&
        threshold_compare_mode_ != "unsigned_promotion") {
        throw std::runtime_error("threshold compare mode 只支持 signed/unsigned_promotion");
    }
    if (neuron_model != "lif" && neuron_model != "st_bif")
        throw std::runtime_error("neuron model 只支持 lif/st_bif");
    if (st_bif_ && (leak_ != 1.0F || tracer_min_ > tracer_max_))
        throw std::runtime_error("ST-BIF leak/tracer 参数无效");
}

SomaNeuronResult SomaState::process_neuron(std::uint64_t neuron, float synaptic_input,
                                           bool has_pending_input, float bias) {
    if (neuron >= voltage_.size()) throw std::runtime_error("soma neuron 越界");
    const auto index = static_cast<std::size_t>(neuron);
    // pending 标志保留“收到过但累加和为 0”的更新；已有膜电位也需要继续做 timestep transition。
    SomaNeuronResult result;
    result.updated = has_pending_input || bias != 0.0F || voltage_[index] != 0.0F;
    if (!result.updated) return result;

    // ST-BIF 配置已强制 leak=1，因此保留 LIF 原有无分支热路径。
    voltage_[index] *= leak_;
    if (membrane_quantization_step_ > 0.0F) {
        voltage_[index] = std::trunc(voltage_[index] / membrane_quantization_step_) *
                          membrane_quantization_step_;
    }
    voltage_[index] += synaptic_input + bias;
    const bool positive_saturation = voltage_[index] > membrane_max_;
    voltage_[index] = std::max(membrane_min_, std::min(membrane_max_, voltage_[index]));
    if (st_bif_) {
        const bool positive = voltage_[index] >= threshold_[index] && tracer_[index] < tracer_max_;
        const bool negative = voltage_[index] < 0.0F && tracer_[index] > tracer_min_;
        if (readout_ || (!positive && !negative)) return result;
        const auto value = positive ? 1.0F : -1.0F;
        voltage_[index] -= value * threshold_[index];
        tracer_[index] += positive ? 1 : -1;
        ++fire_count_[index];
        result.fired = FiredNeuron{neuron, value};
        return result;
    }
    bool threshold_reached = false;
    if (threshold_compare_mode_ == "unsigned_promotion") {
        // 精确对应 C 中 int32 membrane 与 uint32 threshold 比较时的 usual arithmetic conversion。
        const auto membrane_integral = std::trunc(voltage_[index]);
        const auto threshold_integral = std::trunc(threshold_[index]);
        if (!std::isfinite(membrane_integral) || !std::isfinite(threshold_integral) ||
            membrane_integral != voltage_[index] || threshold_integral != threshold_[index] ||
            membrane_integral < static_cast<float>(std::numeric_limits<std::int32_t>::min()) ||
            membrane_integral > static_cast<float>(std::numeric_limits<std::int32_t>::max()) ||
            threshold_integral < 0.0F ||
            threshold_integral > static_cast<float>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("unsigned_promotion 要求 membrane 为 int32、threshold 为 uint32 范围内整数");
        }
        const auto membrane_i32 = static_cast<std::int32_t>(membrane_integral);
        const auto threshold_u32 = static_cast<std::uint32_t>(threshold_integral);
        const auto promoted_membrane = static_cast<std::uint32_t>(membrane_i32);
        threshold_reached = threshold_comparison_ == "greater"
                                ? promoted_membrane > threshold_u32
                                : promoted_membrane >= threshold_u32;
    } else {
        threshold_reached = threshold_comparison_ == "greater"
                                ? voltage_[index] > threshold_[index]
                                : voltage_[index] >= threshold_[index];
    }
    const bool fired = positive_saturation && fire_on_positive_saturation_
                           ? true
                           : threshold_reached;
    if (readout_ || !fired) return result;

    // 每个 neuron 每个 timestep 最多 firing 一次；soft reset 的剩余电位留到后续 timestep。
    result.fired = FiredNeuron{neuron, 1.0F};
    ++fire_count_[index];
    if (positive_saturation && fire_on_positive_saturation_) return result;
    if (reset_ == "soft") voltage_[index] -= threshold_[index];
    else voltage_[index] = 0.0F;
    return result;
}

}  // namespace soma

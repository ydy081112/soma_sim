#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
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
    SomaState(std::size_t neurons, float threshold, float leak, std::string reset, bool readout,
              float membrane_quantization_step = 0.0F,
              std::string threshold_comparison = "greater_equal",
              std::string threshold_compare_mode = "signed",
              std::vector<float> thresholds = {},
              float membrane_min = -std::numeric_limits<float>::infinity(),
              float membrane_max = std::numeric_limits<float>::infinity(),
              bool fire_on_positive_saturation = false,
              std::string neuron_model = "lif", std::int32_t tracer_min = 0,
              std::int32_t tracer_max = 0, std::vector<float> initial_membrane = {});

    SomaNeuronResult process_neuron(std::uint64_t neuron, float synaptic_input,
                                    bool has_pending_input, float bias);
    const std::vector<float>& voltage() const { return voltage_; }
    const std::vector<std::uint32_t>& fire_count() const { return fire_count_; }
    const std::vector<std::int32_t>& tracer() const { return tracer_; }

private:
    std::vector<float> voltage_;
    std::vector<float> threshold_;
    std::vector<std::uint32_t> fire_count_;
    std::vector<std::int32_t> tracer_;
    float leak_ = 1.0F;
    std::string reset_;
    bool readout_ = false;
    float membrane_quantization_step_ = 0.0F;
    std::string threshold_comparison_;
    std::string threshold_compare_mode_;
    float membrane_min_;
    float membrane_max_;
    bool fire_on_positive_saturation_ = false;
    bool st_bif_ = false;
    std::int32_t tracer_min_ = 0, tracer_max_ = 0;
};

}  // namespace soma

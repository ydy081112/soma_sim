#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace soma {

struct SpatialTemplate {
    // plan 选择空间模板，CSR pattern 再给出相对 destination 与 weight offset。
    std::uint32_t cin = 0;
    std::uint32_t cout = 0;
    bool channelwise = false;
    std::vector<float> weight;
    std::vector<std::int32_t> plan_pattern_id;
    std::vector<std::int32_t> plan_dst_base;
    std::vector<std::int32_t> pattern_ptr;
    std::vector<std::int32_t> pattern_dst_offset;
    std::vector<std::int64_t> pattern_weight_offset;

    void validate(std::size_t source_neurons, std::size_t destination_neurons) const;

    template <typename Fn>
    std::uint64_t for_each_destination(std::uint64_t source_neuron, float spike_value, Fn&& fn) const {
        // neuron id 是 spatial-major，因此除法/取模即可拆出空间点和输入通道。
        const auto source_spatial = static_cast<std::size_t>(source_neuron / cin);
        const auto source_channel = static_cast<std::size_t>(source_neuron % cin);
        // 找到它属于哪个 pattern
        const auto pattern = static_cast<std::size_t>(plan_pattern_id.at(source_spatial));
        const auto begin = static_cast<std::size_t>(pattern_ptr.at(pattern));
        const auto end = static_cast<std::size_t>(pattern_ptr.at(pattern + 1));
        std::uint64_t updates = 0;
        for (std::size_t entry = begin; entry < end; ++entry) {
            const auto destination_spatial = static_cast<std::size_t>(
                plan_dst_base[source_spatial] + pattern_dst_offset[entry]);
            if (channelwise) {
                // Pooling 等 channelwise 算子只更新同一通道，不展开 Cout 循环。
                fn(destination_spatial * cout + source_channel,
                   spike_value * weight.at(static_cast<std::size_t>(pattern_weight_offset[entry])));
                ++updates;
                continue;
            }
            const auto weight_base = source_channel * (weight.size() / cin) +
                                     static_cast<std::size_t>(pattern_weight_offset[entry]);
            const auto destination_base = destination_spatial * cout;
            // 固定 source/kernel 后，weight 和 destination 的 Cout 块都连续。
            for (std::size_t output_channel = 0; output_channel < cout; ++output_channel) {
                fn(destination_base + output_channel,
                   spike_value * weight[weight_base + output_channel]);
            }
            updates += cout;
        }
        return updates;
    }
};

}  // namespace soma

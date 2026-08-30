#include "soma/runtime/spatial_template.hpp"

#include <algorithm>
#include <stdexcept>

namespace soma {

void SpatialTemplate::validate(std::size_t source_neurons, std::size_t destination_neurons) const {
    if (cin == 0 || cout == 0) throw std::runtime_error("SpatialTemplate cin/cout 必须为正");
    if (source_neurons % cin != 0 || plan_pattern_id.size() != source_neurons / cin ||
        plan_dst_base.size() != source_neurons / cin) {
        throw std::runtime_error("SpatialTemplate plan 与 source neuron layout 不匹配");
    }
    if (pattern_ptr.empty() || pattern_ptr.front() != 0 ||
        pattern_ptr.back() < 0 || static_cast<std::size_t>(pattern_ptr.back()) != pattern_dst_offset.size() ||
        pattern_dst_offset.size() != pattern_weight_offset.size()) {
        throw std::runtime_error("SpatialTemplate CSR 数组不一致");
    }
    for (const auto pattern : plan_pattern_id) {
        if (pattern < 0 || static_cast<std::size_t>(pattern + 1) >= pattern_ptr.size()) {
            throw std::runtime_error("SpatialTemplate pattern id 越界");
        }
    }
    for (std::size_t source_spatial = 0; source_spatial < plan_pattern_id.size(); ++source_spatial) {
        const auto pattern = static_cast<std::size_t>(plan_pattern_id[source_spatial]);
        for (auto entry = pattern_ptr[pattern]; entry < pattern_ptr[pattern + 1]; ++entry) {
            const auto dst_spatial = static_cast<std::int64_t>(plan_dst_base[source_spatial]) +
                                     pattern_dst_offset[static_cast<std::size_t>(entry)];
            if (dst_spatial < 0 || static_cast<std::size_t>(dst_spatial) * cout >= destination_neurons) {
                throw std::runtime_error("SpatialTemplate destination 越界");
            }
            const auto offset = pattern_weight_offset[static_cast<std::size_t>(entry)];
            if (offset < 0) throw std::runtime_error("SpatialTemplate weight offset 为负");
            if (channelwise) {
                if (static_cast<std::size_t>(offset) >= weight.size() || cin != cout) {
                    throw std::runtime_error("channelwise SpatialTemplate 权重或通道不匹配");
                }
            } else if (weight.size() % cin != 0 || static_cast<std::size_t>(offset) + cout > weight.size() / cin) {
                throw std::runtime_error("SpatialTemplate weight offset 越界");
            }
        }
    }
}

}  // namespace soma


#pragma once

#include "soma/config/mapping_config.hpp"
#include "soma/runtime/spatial_template.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace soma {

struct LayerWeights {
    LayerOp op = LayerOp::Input;
    SpatialTemplate spatial;
    std::vector<float> dense_weight;  // [Cin, Cout]
    std::vector<float> bias;
};

class WeightStore {
public:
    static WeightStore load(const std::string& path, const MappingConfig& mapping);
    const LayerWeights& at(const std::string& layer_id) const;

private:
    std::unordered_map<std::string, LayerWeights> layers_;
};

}  // namespace soma


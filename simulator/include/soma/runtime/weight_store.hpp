#pragma once

#include "soma/config/mapping_config.hpp"
#include "soma/runtime/spatial_template.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace soma {

struct LayerWeights {
    LayerOp op = LayerOp::Input;
    ConnectionType connection_type = ConnectionType::Spatial;
    ConnectionType hardware_type = ConnectionType::Spatial;
    std::uint64_t source_neurons = 0;
    SpatialTemplate spatial;
    std::vector<float> dense_weight;  // [Cin, Cout]
    std::vector<float> identity_weight;  // scalar、逐 channel 或逐 neuron。
    std::vector<float> bias;
};

class WeightStore {
public:
    static WeightStore load(const std::string& path, const MappingConfig& mapping);
    const LayerWeights& at(const std::string& layer_id) const;
    const LayerWeights& connection(std::size_t connection_index) const;

private:
    std::unordered_map<std::string, LayerWeights> layers_;
    std::vector<LayerWeights> connections_;
};

}  // namespace soma

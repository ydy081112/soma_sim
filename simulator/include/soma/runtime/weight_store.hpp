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
    std::vector<float> threshold;
    std::vector<std::uint8_t> active_neuron;
    std::vector<std::uint8_t> crossbar_axon_type;
    std::vector<std::int16_t> crossbar_neuron_weight;  // [physical neuron, weight type]
    std::vector<std::uint64_t> crossbar_rows;  // [Core, axon, 64-neuron word]
    std::uint32_t crossbar_axons = 0;
    std::uint32_t crossbar_words_per_axon = 0;
    std::vector<std::int32_t> route_destination_partition;
    std::vector<std::uint16_t> route_destination_axon;
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

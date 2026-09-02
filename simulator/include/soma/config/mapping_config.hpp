#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace soma {

enum class LayerOp { Input, Conv2d, Linear, AvgPool2d };
enum class ConnectionType { Spatial, Dense, Identity };

struct LayerMapping {
    // 一个条目描述一个 layer partition 到 PE/Core/Router 的静态放置。
    std::size_t index = 0;
    std::string id;
    std::string partition = "0";
    LayerOp op = LayerOp::Input;
    std::uint32_t pe = 0;
    std::uint32_t core = 0;
    std::uint32_t router = 0;
    std::uint64_t neurons = 0;
    std::uint64_t source_neurons = 0;
    std::uint32_t input_h = 1;
    std::uint32_t input_w = 1;
    std::uint32_t input_channels = 1;
    std::uint32_t output_h = 1;
    std::uint32_t output_w = 1;
    std::uint32_t output_channels = 1;
    std::uint32_t aggregate_core_count = 0;
    std::string physical_neuron_order = "logical";
    std::string weight_prefix;
    float threshold = 1.0F;
    float leak = 1.0F;
    std::string reset = "soft";
    float membrane_quantization_step = 0.0F;
    std::string threshold_comparison = "greater_equal";
    bool virtual_input = false;
    bool readout = false;
    bool channelwise = false;

    std::uint64_t physical_neuron_index(std::uint64_t logical_neuron) const;
    std::uint64_t logical_neuron_index(std::uint64_t physical_neuron) const;
};

struct ConnectionMapping {
    std::size_t index = 0;
    std::string from;
    std::string to;
    ConnectionType type = ConnectionType::Spatial;
    ConnectionType hardware_type = ConnectionType::Spatial;
    std::string weight_prefix;
    std::uint32_t delay = 0;
    bool channelwise = false;
};

struct StaticRoute {
    // routers 包含 layer 起点和终点，用于校验/debug；physical packet 使用同样的 XY 语义。
    std::string from;
    std::string to;
    std::vector<std::uint32_t> routers;
};

class MappingConfig {
public:
    std::string model;
    std::vector<LayerMapping> layers;
    std::vector<ConnectionMapping> connections;
    std::vector<StaticRoute> routes;

    static MappingConfig load(const std::string& path);
    void validate(std::uint32_t router_count) const;

    const LayerMapping& layer(const std::string& id) const;
    const LayerMapping& layer(std::size_t index) const { return layers.at(index); }
    const StaticRoute& route(const std::string& from, const std::string& to) const;
    const std::vector<std::size_t>& outgoing(std::size_t layer) const;
    std::size_t index_of(const std::string& id) const;

private:
    std::unordered_map<std::string, std::size_t> layer_index_;
    std::unordered_map<std::string, std::size_t> route_index_;
    std::vector<std::vector<std::size_t>> outgoing_connections_;
    void rebuild_indices();
};

std::string to_string(LayerOp op);
std::string to_string(ConnectionType type);

}  // namespace soma

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace soma {

enum class LayerOp { Input, Conv2d, Linear, AvgPool2d };

struct LayerMapping {
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
    std::string weight_prefix;
    std::string next;
    float threshold = 1.0F;
    float leak = 1.0F;
    std::string reset = "soft";
    bool virtual_input = false;
    bool readout = false;
    bool channelwise = false;
};

struct StaticRoute {
    std::string from;
    std::string to;
    std::vector<std::uint32_t> routers;
};

class MappingConfig {
public:
    std::string model;
    std::vector<LayerMapping> layers;
    std::vector<StaticRoute> routes;

    static MappingConfig load(const std::string& path);
    void validate(std::uint32_t router_count) const;

    const LayerMapping& layer(const std::string& id) const;
    const LayerMapping& layer(std::size_t index) const { return layers.at(index); }
    const StaticRoute& route(const std::string& from, const std::string& to) const;
    std::size_t index_of(const std::string& id) const;

private:
    std::unordered_map<std::string, std::size_t> layer_index_;
    std::unordered_map<std::string, std::size_t> route_index_;
    void rebuild_indices();
};

std::string to_string(LayerOp op);

}  // namespace soma


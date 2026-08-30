#include "soma/config/hardware_config.hpp"

#include "soma/config/simple_yaml.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <limits>
#include <stdexcept>

namespace soma {
namespace {

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) value.erase(value.begin());
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
    return value;
}

SimTime hw_latency_field(const yaml::Node& node, const std::string& key, SimTime fallback) {
    const auto* value = node.find(key);
    return value == nullptr ? fallback : parse_time_ps(value->as_string());
}

const yaml::Node& optional_map(const yaml::Node& parent, const std::string& key) {
    static const yaml::Node empty = yaml::Node::map();
    const auto* value = parent.find(key);
    return value == nullptr ? empty : *value;
}

}  // namespace

SimTime parse_time_ps(const std::string& raw) {
    // 配置加载时一次性换算为整数 ps，事件热路径不再处理单位或浮点时间。
    auto value = trim(raw);
    std::size_t consumed = 0;
    const long double magnitude = std::stold(value, &consumed);
    auto unit = trim(value.substr(consumed));
    std::transform(unit.begin(), unit.end(), unit.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    long double scale = 1.0L;
    if (unit.empty() || unit == "ps") scale = 1.0L;
    else if (unit == "ns") scale = static_cast<long double>(kPsPerNs);
    else if (unit == "us") scale = static_cast<long double>(kPsPerUs);
    else if (unit == "ms") scale = static_cast<long double>(kPsPerMs);
    else if (unit == "s") scale = static_cast<long double>(kPsPerSecond);
    else throw std::runtime_error("不支持的时间单位: " + raw);
    const long double ps = magnitude * scale;
    if (ps < 0.0L || ps > static_cast<long double>(std::numeric_limits<SimTime>::max())) {
        throw std::runtime_error("时间超出 SimTime 范围: " + raw);
    }
    return static_cast<SimTime>(std::llround(ps));
}

HardwareConfig HardwareConfig::load(const std::string& path) {
    const auto root = yaml::load_file(path);
    const auto& architecture = root.at("architecture");
    HardwareConfig config;
    config.name = yaml::get_string(architecture, "name", "configurable_snn");
    config.execution_mode = architecture.at("execution_mode").as_string();
    config.frequency_mhz = yaml::get_double(architecture, "frequency_mhz", 1'000.0);
    config.hw_cycle_time_ps = static_cast<SimTime>(
        std::llround(1'000'000.0 / config.frequency_mhz));

    // NoC、Core 和能耗分别加载，缺省值只用于可选微架构字段。
    const auto& noc = architecture.at("noc");
    config.noc.topology = yaml::get_string(noc, "topology", "mesh");
    config.noc.routing = yaml::get_string(noc, "routing", "static");
    config.noc.rows = static_cast<std::uint32_t>(yaml::get_u64(noc, "rows", 1));
    config.noc.cols = static_cast<std::uint32_t>(yaml::get_u64(noc, "cols", 1));
    config.noc.virtual_channels = static_cast<std::uint32_t>(yaml::get_u64(noc, "virtual_channels", 1));
    config.noc.input_buffer_depth = static_cast<std::uint32_t>(yaml::get_u64(noc, "input_buffer_depth", 1));
    config.noc.flit_width_bits = static_cast<std::uint32_t>(yaml::get_u64(noc, "flit_width_bits", 64));
    config.noc.spike_per_flit = static_cast<std::uint32_t>(yaml::get_u64(noc, "spike_per_flit", 1));

    const auto& pipeline = optional_map(noc, "pipeline");
    config.noc.input_queue_hw_latency = hw_latency_field(
        pipeline, "input_queue_hardware_latency", 0);
    config.noc.route_compute_hw_latency = hw_latency_field(
        pipeline, "route_compute_hardware_latency", 0);
    config.noc.switch_allocate_hw_latency = hw_latency_field(
        pipeline, "switch_allocate_hardware_latency", 0);
    config.noc.switch_traversal_hw_latency = hw_latency_field(
        pipeline, "switch_traversal_hardware_latency", 0);
    const auto& link = noc.at("link");
    config.noc.link_hw_latency = hw_latency_field(link, "hardware_latency", 0);
    const auto& directional_hw_latency = optional_map(link, "directional_hardware_latency");
    config.noc.north_link_hw_latency = hw_latency_field(
        directional_hw_latency, "north", config.noc.link_hw_latency);
    config.noc.east_link_hw_latency = hw_latency_field(
        directional_hw_latency, "east", config.noc.link_hw_latency);
    config.noc.south_link_hw_latency = hw_latency_field(
        directional_hw_latency, "south", config.noc.link_hw_latency);
    config.noc.west_link_hw_latency = hw_latency_field(
        directional_hw_latency, "west", config.noc.link_hw_latency);
    config.noc.link_busy_hw_latency = hw_latency_field(
        link, "busy_hardware_latency", config.noc.link_hw_latency);
    // Link 的同步/异步模式由信号线组合决定，不依赖 architecture.name。
    const auto& send = optional_map(link, "send");
    const auto& receive = optional_map(link, "receive");
    config.noc.send_req = yaml::get_bool(send, "req", false);
    config.noc.send_data = yaml::get_bool(send, "data", true);
    config.noc.receive_ack = yaml::get_bool(receive, "ack", false);
    config.noc.receive_credit = yaml::get_bool(receive, "credit", false);

    const auto& core = architecture.at("core");
    config.core.pe_count = static_cast<std::uint32_t>(yaml::get_u64(core, "pe_count", 1));
    config.core.cores_per_pe = static_cast<std::uint32_t>(yaml::get_u64(core, "cores_per_pe", 1));
    config.core.max_neurons = static_cast<std::uint32_t>(yaml::get_u64(core, "max_neurons", 1'024));
    config.core.input_buffer_depth = static_cast<std::uint32_t>(yaml::get_u64(core, "input_buffer_depth", 16));
    config.core.synapse_sram_bytes = yaml::get_u64(core, "synapse_sram_bytes", 128 * 1024);
    const auto& hw_latency = core.at("hardware_latency");
    config.core.axon_in_hw_latency = hw_latency_field(hw_latency, "axon_in", 0);
    config.core.axon_out_hw_latency = hw_latency_field(hw_latency, "axon_out", 0);
    config.core.sram_read_hw_latency = hw_latency_field(hw_latency, "sram_read", 0);
    config.core.sram_write_hw_latency = hw_latency_field(hw_latency, "sram_write", 0);
    config.core.synapse_hw_latency = hw_latency_field(hw_latency, "synapse", 0);
    config.core.soma_access_hw_latency = hw_latency_field(hw_latency, "soma_access", 0);
    config.core.soma_update_hw_latency = hw_latency_field(hw_latency, "soma_update", 0);
    config.core.soma_fire_hw_latency = hw_latency_field(hw_latency, "soma_fire", 0);
    const auto& neuron = optional_map(core, "neuron");
    config.core.default_threshold = static_cast<float>(yaml::get_double(neuron, "threshold", 1.0));
    config.core.default_leak = static_cast<float>(yaml::get_double(neuron, "leak", 1.0));
    config.core.reset = yaml::get_string(neuron, "reset", "soft");

    const auto& energy = architecture.at("energy_pj");
    config.energy.router_hop_pj = yaml::get_double(energy, "router_hop", 0.0);
    config.energy.link_hop_pj = yaml::get_double(energy, "link_hop", 0.0);
    const auto& directional_energy = optional_map(energy, "directional_link");
    config.energy.north_link_pj = yaml::get_double(directional_energy, "north", config.energy.link_hop_pj);
    config.energy.east_link_pj = yaml::get_double(directional_energy, "east", config.energy.link_hop_pj);
    config.energy.south_link_pj = yaml::get_double(directional_energy, "south", config.energy.link_hop_pj);
    config.energy.west_link_pj = yaml::get_double(directional_energy, "west", config.energy.link_hop_pj);
    config.energy.axon_in_pj = yaml::get_double(energy, "axon_in", 0.0);
    config.energy.axon_out_pj = yaml::get_double(energy, "axon_out", 0.0);
    config.energy.sram_read_pj = yaml::get_double(energy, "sram_read", 0.0);
    config.energy.sram_write_pj = yaml::get_double(energy, "sram_write", 0.0);
    config.energy.synapse_pj = yaml::get_double(energy, "synapse", 0.0);
    config.energy.soma_update_pj = yaml::get_double(energy, "soma_update", 0.0);
    config.energy.soma_fire_pj = yaml::get_double(energy, "soma_fire", 0.0);
    config.validate();
    return config;
}

void HardwareConfig::validate() const {
    // 在进入仿真循环前拒绝当前 MVP 无法准确表达的硬件组合。
    if (!timestep_synchronization()) {
        throw std::runtime_error(
            "当前版本只实现 execution_mode: timestep_synchronization");
    }
    if (frequency_mhz <= 0.0 || hw_cycle_time_ps == 0) {
        throw std::runtime_error("frequency_mhz 必须为正");
    }
    if (noc.topology != "mesh") throw std::runtime_error("MVP 目前只支持 mesh topology");
    if (noc.rows == 0 || noc.cols == 0) throw std::runtime_error("NoC rows/cols 必须为正");
    if (noc.virtual_channels != 1) throw std::runtime_error("MVP 只支持一个 virtual channel");
    if (!noc.send_data) throw std::runtime_error("link.send.data 必须启用");
    if (noc.send_req != noc.receive_ack) {
        throw std::runtime_error("异步 link 必须同时配置 send.req 与 receive.ack");
    }
    if (core.pe_count == 0 || core.max_neurons == 0 || core.cores_per_pe == 0) {
        throw std::runtime_error("PE/Core 容量必须为正");
    }
    if (core.default_threshold <= 0.0F) throw std::runtime_error("默认 threshold 必须为正");
}

}  // namespace soma

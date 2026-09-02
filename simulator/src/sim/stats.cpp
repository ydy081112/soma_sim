#include "soma/sim/stats.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace soma {
namespace {

double host_rate(std::uint64_t count, double host_latency_s) {
    return host_latency_s > 0.0 ? static_cast<double>(count) / host_latency_s : 0.0;
}

int prediction(const std::vector<float>& scores) {
    if (scores.empty()) return -1;
    return static_cast<int>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
}

}  // namespace

Statistics::Statistics(const MappingConfig& mapping, const HardwareConfig& hardware)
    : mapping_(mapping), hardware_(hardware), layers_(mapping.layers.size()) {}

TimestepStats& Statistics::timestep(std::uint32_t index) {
    // timestep 只在首次出现时扩展，避免预先按最大时步分配大表。
    if (timesteps_.size() <= index) timesteps_.resize(static_cast<std::size_t>(index) + 1);
    return timesteps_[index];
}

void Statistics::set_physical_topology(std::uint64_t physical_cores,
                                       std::uint64_t mapped_tiles) {
    physical_core_count_ = physical_cores;
    mapped_tile_count_ = mapped_tiles;
}

void Statistics::begin_timestep(std::uint32_t step, SimTime hw_start_time) {
    auto& metric = timestep(step);
    metric.hw_start_time = hw_start_time;
    metric.hw_end_time = std::max(metric.hw_end_time, hw_start_time);
}

void Statistics::complete_timestep(std::uint32_t step, SimTime hw_end_time,
                                   SimTime synchronization_hw_latency) {
    auto& metric = timestep(step);
    metric.synchronization_hw_latency += synchronization_hw_latency;
    metric.hw_end_time = std::max(metric.hw_end_time, hw_end_time);
    breakdown_.synchronization_hw_latency += synchronization_hw_latency;
    hw_latency_ = std::max(hw_latency_, hw_end_time);
}

void Statistics::record_packet(std::size_t layer, std::uint32_t step, std::uint64_t updates,
                               const NocTiming& noc, SimTime hw_current_time) {
    ++processed_spikes_;
    ++packets_;
    noc_hops_ += noc.hops;
    synaptic_updates_ += updates;
    ++layers_.at(layer).processed_spikes;
    ++layers_.at(layer).packets;
    layers_.at(layer).noc_hops += noc.hops;
    layers_.at(layer).synaptic_updates += updates;
    auto& metric = timestep(step);
    ++metric.processed_spikes;
    ++metric.packets;
    metric.noc_hops += noc.hops;
    metric.synaptic_updates += updates;
    metric.hw_end_time = std::max(metric.hw_end_time, hw_current_time);
    hw_latency_ = std::max(hw_latency_, hw_current_time);
}

void Statistics::record_emit(std::size_t layer, std::uint32_t step, SimTime hw_current_time) {
    ++emitted_spikes_;
    ++layers_.at(layer).emitted_spikes;
    auto& metric = timestep(step);
    ++metric.emitted_spikes;
    metric.hw_end_time = std::max(metric.hw_end_time, hw_current_time);
    hw_latency_ = std::max(hw_latency_, hw_current_time);
}

void Statistics::record_neuron_processing(std::size_t layer, std::uint32_t step,
                                          SimTime hw_current_time) {
    static_cast<void>(layer);
    auto& metric = timestep(step);
    metric.hw_end_time = std::max(metric.hw_end_time, hw_current_time);
    hw_latency_ = std::max(hw_latency_, hw_current_time);
}

void Statistics::record_host_latency(std::size_t layer, std::uint32_t step,
                                     double host_latency_s) {
    layers_.at(layer).host_latency_s += host_latency_s;
    timestep(step).host_latency_s += host_latency_s;
}

void Statistics::record_timestep_host_latency(std::uint32_t step, double host_latency_s) {
    // timestep CSV 保留整个 timestep 的 host 聚合，不再依赖逐 packet 时钟采样。
    timestep(step).host_latency_s = host_latency_s;
}

void Statistics::add_inject_hw_latency(std::uint32_t, SimTime hw_latency) {
    breakdown_.pe_inject_hw_latency += hw_latency;
}
void Statistics::add_noc_hw_latency(std::uint32_t step, const NocTiming& timing) {
    // traversal、congestion 和 link busy 分开累计，便于定位带宽瓶颈。
    breakdown_.noc_traversal_hw_latency += timing.hw_traversal_latency;
    breakdown_.router_congestion_hw_latency += timing.hw_congestion_latency;
    breakdown_.link_busy_hw_latency += timing.hw_link_busy_latency;
    auto& metric = timestep(step);
    metric.noc_traversal_hw_latency += timing.hw_traversal_latency;
    metric.noc_congestion_hw_latency += timing.hw_congestion_latency;
}
void Statistics::add_synapse_hw_latency(std::uint32_t step, SimTime service_hw_latency,
                                        SimTime total_hw_latency) {
    breakdown_.synapse_service_hw_latency += service_hw_latency;
    breakdown_.pe_compute_hw_latency += total_hw_latency;
    timestep(step).synapse_service_hw_latency += service_hw_latency;
}
void Statistics::add_soma_hw_latency(std::uint32_t step, SimTime service_hw_latency,
                                     SimTime total_hw_latency) {
    breakdown_.soma_service_hw_latency += service_hw_latency;
    breakdown_.pe_compute_hw_latency += total_hw_latency;
    timestep(step).soma_service_hw_latency += service_hw_latency;
}

void Statistics::add_data_energy(const NocTiming& noc, std::uint64_t updates,
                                 ConnectionType connection_type) {
    // 通信能耗按方向 hop 计数，计算能耗按实际 synaptic update 计数。
    energy_.axon_pj += hardware_.energy.axon_out_pj + hardware_.energy.axon_in_pj;
    energy_.router_pj += static_cast<double>(noc.hops + 1) * hardware_.energy.router_hop_pj;
    energy_.link_pj += static_cast<double>(noc.port_hops[static_cast<std::size_t>(Port::North)]) * hardware_.energy.north_link_pj +
                       static_cast<double>(noc.port_hops[static_cast<std::size_t>(Port::East)]) * hardware_.energy.east_link_pj +
                       static_cast<double>(noc.port_hops[static_cast<std::size_t>(Port::South)]) * hardware_.energy.south_link_pj +
                       static_cast<double>(noc.port_hops[static_cast<std::size_t>(Port::West)]) * hardware_.energy.west_link_pj;
    energy_.memory_pj += hardware_.energy.sram_read_pj +
                         static_cast<double>(updates) * hardware_.energy.sram_write_pj;
    const auto synapse_pj = connection_type == ConnectionType::Dense
                                ? hardware_.energy.dense_synapse_pj
                            : connection_type == ConnectionType::Identity
                                ? hardware_.energy.identity_synapse_pj
                                : hardware_.energy.spatial_synapse_pj;
    energy_.synapse_pj += static_cast<double>(updates) * synapse_pj;
}

void Statistics::add_neuron_energy(std::uint64_t updated_neurons) {
    energy_.soma_pj += static_cast<double>(updated_neurons) * hardware_.energy.soma_update_pj;
}

void Statistics::add_fire_energy() {
    energy_.soma_pj += hardware_.energy.soma_fire_pj;
}

void Statistics::write(const std::string& output_dir, const std::vector<float>& scores,
                       std::optional<int> expected_output) const {
    // summary 面向单次结果，两个 CSV 分别保留逐层和逐 timestep 的可分析数据。
    std::filesystem::create_directories(output_dir);
    std::ofstream summary(std::filesystem::path(output_dir) / "summary.json");
    if (!summary) throw std::runtime_error("无法写 output/summary.json");
    summary << std::setprecision(12);
    summary << "{\n"
            << "  \"model\": \"" << mapping_.model << "\",\n"
            << "  \"completed\": " << (stopped_early_ ? "false" : "true") << ",\n"
            << "  \"hardware_latency_ps\": " << hw_latency_ << ",\n"
            << "  \"hardware_latency_s\": " << static_cast<double>(hw_latency_) / kPsPerSecond << ",\n"
            << "  \"physical_core_count\": " << physical_core_count_ << ",\n"
            << "  \"mapped_tile_count\": " << mapped_tile_count_ << ",\n"
            << "  \"total_spikes\": " << processed_spikes_ << ",\n"
            << "  \"total_emitted_spikes\": " << emitted_spikes_ << ",\n"
            << "  \"packets\": " << packets_ << ",\n"
            << "  \"noc_hops\": " << noc_hops_ << ",\n"
            << "  \"synaptic_updates\": " << synaptic_updates_ << ",\n"
            << "  \"host_latency_s\": " << host_latency_s_ << ",\n"
            << "  \"host_processed_spikes_per_sec\": "
            << host_rate(processed_spikes_, host_latency_s_) << ",\n"
            << "  \"hardware_latency_breakdown_cycles\": {\n"
            << "    \"pe_inject_cycles\": " << ceil_cycles(
                   breakdown_.pe_inject_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"pe_compute_cycles\": " << ceil_cycles(
                   breakdown_.pe_compute_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"soma_service_cycles\": " << ceil_cycles(
                   breakdown_.soma_service_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"synapse_service_cycles\": " << ceil_cycles(
                   breakdown_.synapse_service_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"noc_traversal_cycles\": " << ceil_cycles(
                   breakdown_.noc_traversal_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"router_congestion_cycles\": " << ceil_cycles(
                   breakdown_.router_congestion_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"link_busy_cycles\": " << ceil_cycles(
                   breakdown_.link_busy_hw_latency, hardware_.hw_cycle_time_ps) << ",\n"
            << "    \"timestep_synchronization_cycles\": " << ceil_cycles(
                   breakdown_.synchronization_hw_latency, hardware_.hw_cycle_time_ps) << "\n"
            << "  },\n"
            << "  \"energy_pj\": {\n"
            << "    \"axon\": " << energy_.axon_pj << ",\n"
            << "    \"router\": " << energy_.router_pj << ",\n"
            << "    \"link\": " << energy_.link_pj << ",\n"
            << "    \"memory\": " << energy_.memory_pj << ",\n"
            << "    \"synapse\": " << energy_.synapse_pj << ",\n"
            << "    \"soma\": " << energy_.soma_pj << ",\n"
            << "    \"total\": " << energy_.total_pj() << "\n"
            << "  },\n"
            << "  \"output_scores\": [";
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (i != 0) summary << ", ";
        summary << scores[i];
    }
    summary << "],\n  \"prediction\": " << prediction(scores) << ",\n  \"expected_output\": ";
    if (expected_output) summary << *expected_output;
    else summary << "null";
    summary << "\n}\n";

    std::ofstream layer_csv(std::filesystem::path(output_dir) / "layer_metrics.csv");
    layer_csv << "layer_id,processed_spikes,emitted_spikes,packets,noc_hops,synaptic_updates,"
                 "host_latency_s,host_processed_spikes_per_sec\n";
    layer_csv << std::setprecision(12);
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const auto& metric = layers_[i];
        layer_csv << mapping_.layers[i].id << ',' << metric.processed_spikes << ','
                  << metric.emitted_spikes << ',' << metric.packets << ',' << metric.noc_hops << ','
                  << metric.synaptic_updates << ','
                  << metric.host_latency_s << ','
                  << host_rate(metric.processed_spikes, metric.host_latency_s) << '\n';
    }

    std::ofstream timestep_csv(std::filesystem::path(output_dir) / "timestep_metrics.csv");
    timestep_csv << "timestep,processed_spikes,emitted_spikes,packets,noc_hops,synaptic_updates,"
                    "host_latency_s,host_processed_spikes_per_sec,hardware_latency_ps,"
                    "hardware_end_time_ps,soma_service_cycles,synapse_service_cycles,"
                    "noc_traversal_cycles,noc_congestion_cycles,timestep_synchronization_cycles\n";
    timestep_csv << std::setprecision(12);
    // timestep synchronization 的逻辑编号从 1 开始，不输出虚构的 timestep 0。
    const std::size_t first_timestep = hardware_.timestep_synchronization() ? 1 : 0;
    for (std::size_t i = first_timestep; i < timesteps_.size(); ++i) {
        const auto& metric = timesteps_[i];
        timestep_csv << i << ',' << metric.processed_spikes << ',' << metric.emitted_spikes << ','
                     << metric.packets << ',' << metric.noc_hops << ',' << metric.synaptic_updates << ','
                     << metric.host_latency_s << ','
                     << host_rate(metric.processed_spikes, metric.host_latency_s) << ','
                     << (metric.hw_end_time - metric.hw_start_time) << ',' << metric.hw_end_time << ','
                     << ceil_cycles(metric.soma_service_hw_latency, hardware_.hw_cycle_time_ps) << ','
                     << ceil_cycles(metric.synapse_service_hw_latency, hardware_.hw_cycle_time_ps) << ','
                     << ceil_cycles(metric.noc_traversal_hw_latency, hardware_.hw_cycle_time_ps) << ','
                     << ceil_cycles(metric.noc_congestion_hw_latency, hardware_.hw_cycle_time_ps) << ','
                     << ceil_cycles(metric.synchronization_hw_latency, hardware_.hw_cycle_time_ps) << '\n';
    }
}

}  // namespace soma

#pragma once

#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/noc/router.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace soma {

struct LayerStats {
    std::uint64_t processed_spikes = 0;
    std::uint64_t emitted_spikes = 0;
    std::uint64_t packets = 0;
    std::uint64_t noc_hops = 0;
    std::uint64_t synaptic_updates = 0;
    double host_latency_s = 0.0;
};

struct TimestepStats {
    std::uint64_t processed_spikes = 0;
    std::uint64_t emitted_spikes = 0;
    std::uint64_t neuron_firings = 0;
    std::uint64_t packets = 0;
    std::uint64_t noc_hops = 0;
    std::uint64_t synaptic_updates = 0;
    double host_latency_s = 0.0;
    SimTime hw_start_time = 0;
    SimTime hw_end_time = 0;
    SimTime soma_service_hw_latency = 0;
    SimTime synapse_service_hw_latency = 0;
    SimTime noc_traversal_hw_latency = 0;
    SimTime noc_congestion_hw_latency = 0;
    SimTime synchronization_hw_latency = 0;
};

struct BreakdownStats {
    // 这些是所有事件的硬件 latency 累加，不等于关键路径的 hw_end_time。
    SimTime pe_inject_hw_latency = 0;
    SimTime pe_compute_hw_latency = 0;
    SimTime soma_service_hw_latency = 0;
    SimTime synapse_service_hw_latency = 0;
    SimTime noc_traversal_hw_latency = 0;
    SimTime router_congestion_hw_latency = 0;
    SimTime link_busy_hw_latency = 0;
    SimTime synchronization_hw_latency = 0;
};

struct EnergyStats {
    double axon_pj = 0.0;
    double router_pj = 0.0;
    double link_pj = 0.0;
    double memory_pj = 0.0;
    double synapse_pj = 0.0;
    double soma_pj = 0.0;
    double total_pj() const { return axon_pj + router_pj + link_pj + memory_pj + synapse_pj + soma_pj; }
};

struct FiringTraceRecord {
    std::uint32_t timestep = 0;
    std::uint32_t core = 0;
    std::uint32_t local_neuron = 0;
};

class Statistics {
public:
    // host latency 衡量模拟器运行性能，hardware latency 衡量被模拟架构性能。
    Statistics(const MappingConfig& mapping, const HardwareConfig& hardware);

    void set_physical_topology(std::uint64_t physical_cores, std::uint64_t mapped_tiles);
    void begin_timestep(std::uint32_t timestep, SimTime hw_start_time);
    void complete_timestep(std::uint32_t timestep, SimTime hw_end_time,
                           SimTime synchronization_hw_latency);
    void record_packet(std::size_t layer, std::uint32_t timestep, std::uint64_t updates,
                       const NocTiming& noc, SimTime hw_current_time);
    void record_emit(std::size_t layer, std::uint32_t timestep, SimTime hw_current_time);
    void record_neuron_fire(std::size_t layer, std::uint32_t timestep,
                            SimTime hw_current_time, std::uint32_t global_core,
                            std::uint32_t local_neuron);
    void record_neuron_processing(std::size_t layer, std::uint32_t timestep,
                                  SimTime hw_current_time);
    void record_host_latency(std::size_t layer, std::uint32_t timestep, double host_latency_s);
    void record_timestep_host_latency(std::uint32_t timestep, double host_latency_s);
    void add_inject_hw_latency(std::uint32_t timestep, SimTime hw_latency);
    void add_noc_hw_latency(std::uint32_t timestep, const NocTiming& timing);
    void add_synapse_hw_latency(std::uint32_t timestep, SimTime service_hw_latency,
                                SimTime total_hw_latency);
    void add_soma_hw_latency(std::uint32_t timestep, SimTime service_hw_latency,
                             SimTime total_hw_latency);
    void add_data_energy(const NocTiming& noc, std::uint64_t updates,
                         ConnectionType connection_type);
    void add_neuron_energy(std::uint64_t updated_neurons);
    void add_fire_energy();
    void set_host_latency(double host_latency_s) { host_latency_s_ = host_latency_s; }
    void set_stopped_early(bool value) { stopped_early_ = value; }

    void write(const std::string& output_dir, const std::vector<float>& scores,
               std::optional<int> expected_output) const;
    SimTime hw_latency() const { return hw_latency_; }
    double host_latency_s() const { return host_latency_s_; }
    std::uint64_t processed_spikes() const { return processed_spikes_; }

private:
    const MappingConfig& mapping_;
    const HardwareConfig& hardware_;
    std::vector<LayerStats> layers_;
    std::vector<TimestepStats> timesteps_;
    BreakdownStats breakdown_;
    EnergyStats energy_;
    SimTime hw_latency_ = 0;
    std::uint64_t processed_spikes_ = 0;
    std::uint64_t emitted_spikes_ = 0;
    std::uint64_t neuron_firings_ = 0;
    std::vector<std::uint64_t> reported_neuron_firings_;
    std::vector<FiringTraceRecord> firing_trace_;
    std::uint64_t packets_ = 0;
    std::uint64_t noc_hops_ = 0;
    std::uint64_t synaptic_updates_ = 0;
    std::uint64_t physical_core_count_ = 0;
    std::uint64_t mapped_tile_count_ = 0;
    double host_latency_s_ = 0.0;
    bool stopped_early_ = false;

    TimestepStats& timestep(std::uint32_t index);
};

}  // namespace soma

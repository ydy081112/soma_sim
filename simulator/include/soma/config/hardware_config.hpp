#pragma once

#include "soma/common/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace soma {

struct EnergyConfig {
    double router_hop_pj = 0.0;
    double link_hop_pj = 0.0;
    double north_link_pj = 0.0;
    double east_link_pj = 0.0;
    double south_link_pj = 0.0;
    double west_link_pj = 0.0;
    double axon_in_pj = 0.0;
    double axon_out_pj = 0.0;
    double sram_read_pj = 0.0;
    double sram_write_pj = 0.0;
    double synapse_pj = 0.0;
    double soma_update_pj = 0.0;
    double soma_fire_pj = 0.0;
};

struct HardwareConfig {
    std::string name;
    double frequency_mhz = 1'000.0;
    SimTime hw_cycle_time_ps = 1'000;

    struct Noc {
        std::string topology = "mesh";
        std::string routing = "static";
        std::uint32_t rows = 1;
        std::uint32_t cols = 1;
        std::uint32_t virtual_channels = 1;
        std::uint32_t input_buffer_depth = 1;
        std::uint32_t flit_width_bits = 64;
        std::uint32_t spike_per_flit = 1;

        SimTime input_queue_hw_latency = 0;
        SimTime route_compute_hw_latency = 0;
        SimTime switch_allocate_hw_latency = 0;
        SimTime switch_traversal_hw_latency = 0;
        SimTime link_hw_latency = 0;
        SimTime north_link_hw_latency = 0;
        SimTime east_link_hw_latency = 0;
        SimTime south_link_hw_latency = 0;
        SimTime west_link_hw_latency = 0;
        SimTime link_busy_hw_latency = 0;

        bool send_req = false;
        bool send_data = true;
        bool receive_ack = false;
        bool receive_credit = false;

        bool asynchronous() const { return send_req && receive_ack; }
        SimTime router_hw_latency() const {
            return input_queue_hw_latency + route_compute_hw_latency +
                   switch_allocate_hw_latency + switch_traversal_hw_latency;
        }
        std::size_t router_count() const {
            return static_cast<std::size_t>(rows) * cols;
        }
    } noc;

    struct Core {
        std::uint32_t pe_count = 1;
        std::uint32_t cores_per_pe = 1;
        std::uint32_t max_neurons = 1'024;
        std::uint32_t input_buffer_depth = 16;
        std::uint64_t synapse_sram_bytes = 128 * 1024;
        SimTime axon_in_hw_latency = 0;
        SimTime axon_out_hw_latency = 0;
        SimTime sram_read_hw_latency = 0;
        SimTime sram_write_hw_latency = 0;
        SimTime synapse_hw_latency = 0;
        SimTime soma_update_hw_latency = 0;
        SimTime soma_fire_hw_latency = 0;
        float default_threshold = 1.0F;
        float default_leak = 1.0F;
        std::string reset = "soft";
    } core;

    EnergyConfig energy;

    static HardwareConfig load(const std::string& path);
    void validate() const;
};

}  // namespace soma

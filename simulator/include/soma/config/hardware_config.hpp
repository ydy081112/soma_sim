#pragma once

#include "soma/common/types.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <limits>
#include <string>

namespace soma {

struct EnergyConfig {
    // 全部能耗使用 pJ，按事件次数累计；这里不包含 leakage 的时间积分。
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
    double spatial_synapse_pj = 0.0;
    double dense_synapse_pj = 0.0;
    double identity_synapse_pj = 0.0;
    double crossbar_synapse_pj = 0.0;
    double soma_update_pj = 0.0;
    double soma_fire_pj = 0.0;
};

struct HardwareConfig {
    // hardware latency 与 host latency 严格分离；所有硬件时间在加载后均为整数 ps。
    std::string name;
    std::string execution_mode;
    double frequency_mhz = 1'000.0;
    SimTime hw_cycle_time_ps = 1'000;

    struct Noc {
        std::string topology = "mesh";
        std::string routing = "static";
        // resource_table/destination_flow 保留既有时序；route_density 为可选拥塞模型。
        std::string congestion_model = "destination_flow";
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

        // key 是参与同步的 tile 数下界，查询时沿用 SANA-FE 的向下取整表语义。
        std::map<std::uint32_t, SimTime> timestep_sync_hw_latency;

        // req+ack 同时存在时按异步握手资源占用建模。
        bool asynchronous() const { return send_req && receive_ack; }
        SimTime router_hw_latency() const {
            return input_queue_hw_latency + route_compute_hw_latency +
                   switch_allocate_hw_latency + switch_traversal_hw_latency;
        }
        std::size_t router_count() const {
            return static_cast<std::size_t>(rows) * cols;
        }
        SimTime synchronization_hw_latency(std::size_t mapped_tiles) const;
    } noc;

    struct Core {
        std::uint32_t pe_count = 1;
        std::uint32_t cores_per_pe = 1;
        std::uint32_t max_neurons = 1'024;
        std::uint32_t max_axons = 1'024;
        std::uint32_t cores_per_tile = 4;
        std::uint32_t cores_per_chip = 4'096;
        std::uint32_t input_buffer_depth = 16;
        bool input_fifo = false;
        bool fifo_per_core = false;
        // 启用后 global queue 中每个 source Core 只保留一个队首 packet。
        bool source_packet_fifo = false;
        std::uint32_t fifo_num_per_core = 1;
        std::uint32_t fifo_depth_per_core = 1;
        std::uint64_t synapse_sram_bytes = 128 * 1024;
        SimTime axon_in_hw_latency = 0;
        SimTime axon_out_hw_latency = 0;
        SimTime sram_read_hw_latency = 0;
        SimTime sram_write_hw_latency = 0;
        SimTime spatial_synapse_hw_latency = 0;
        SimTime dense_synapse_hw_latency = 0;
        SimTime identity_synapse_hw_latency = 0;
        SimTime crossbar_synapse_hw_latency = 0;
        SimTime soma_access_hw_latency = 0;
        SimTime soma_update_hw_latency = 0;
        SimTime soma_fire_hw_latency = 0;
        float default_threshold = 1.0F;
        float default_leak = 1.0F;
        std::string reset = "soft";
        // all_mapped 保持既有同步扫描；event_activated_catch_up 复现按事件启动的 heartbeat。
        std::string neuron_update_mode = "all_mapped";
        // signed 使用配置权重；nonzero_binary 将任意非零权重解释为单位脉冲贡献。
        std::string crossbar_weight_mode = "signed";
        // signed 保持常规有符号阈值比较；unsigned_promotion 复现 int32/uint32 的 C 提升。
        std::string threshold_compare_mode = "signed";
        // 某些 crossbar 将一个 axon packet 广播给 Core 内全部 neuron heartbeat。
        bool crossbar_packet_activates_all_neurons = false;
        // 兼容会把未映射 crossbar slot 也作为零初始化 neuron 执行的事件模型。
        bool process_inactive_neurons_on_crossbar_event = false;
        float membrane_min = -std::numeric_limits<float>::infinity();
        float membrane_max = std::numeric_limits<float>::infinity();
        bool fire_on_positive_saturation = false;
    } core;

    struct StatisticsConfig {
        std::uint32_t firing_timestep_offset = 0;
        // 默认不保存逐 neuron trace，避免大规模 benchmark 的 I/O 与内存开销。
        bool firing_trace = false;
    } statistics;

    EnergyConfig energy;

    bool timestep_synchronization() const {
        return execution_mode == "timestep_synchronization";
    }
    static HardwareConfig load(const std::string& path);
    void validate() const;
};

}  // namespace soma

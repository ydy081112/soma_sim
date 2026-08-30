#pragma once

#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/buffer.hpp"
#include "soma/hw/core.hpp"
#include "soma/hw/noc/router.hpp"
#include "soma/input_encoder.hpp"
#include "soma/runtime/weight_store.hpp"
#include "soma/sim/spike_queue.hpp"
#include "soma/sim/stats.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace soma {

struct SimulatorOptions {
    std::string hardware_path = "arch/hardware.yaml";
    std::string mapping_path = "compiler/mapping_output/mapping.yaml";
    std::string weights_path = "input/weights.npz";
    std::string input_spikes_path = "input/input_spike.csv";
    std::string output_dir = "output";
    std::uint64_t max_events = 0;  // 0 表示不限制。
};

struct SimulationResult {
    SimTime hw_latency_ps = 0;
    double host_latency_s = 0.0;
    std::uint64_t processed_spikes = 0;
    bool completed = true;
    std::vector<float> output_scores;
};

class Simulator {
public:
    explicit Simulator(SimulatorOptions options);
    SimulationResult run();

private:
    SimulatorOptions options_;
    HardwareConfig hardware_;
    MappingConfig mapping_;
    WeightStore weights_;
    InputSpikeFile input_;
    SpikeQueue queue_;
    RouterResourceTable routers_;
    std::vector<std::unique_ptr<Core>> cores_;
    std::vector<BufferResource> injection_ports_;
    Statistics stats_;

    void load_input_queue();
    void process_data(Spike& spike);
    void process_bias(Spike& spike);
    void process_drain(Spike& spike);
    void schedule_drain(std::size_t layer, std::uint32_t timestep, SimTime hw_time);
    const std::vector<float>& final_scores() const;
};

}  // namespace soma

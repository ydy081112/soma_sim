#pragma once

#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/hardware_resource.hpp"
#include "soma/hw/core.hpp"
#include "soma/hw/noc/router.hpp"
#include "soma/input_encoder.hpp"
#include "soma/runtime/weight_store.hpp"
#include "soma/sim/spike_queue.hpp"
#include "soma/sim/stats.hpp"

#include <cstdint>
#include <map>
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
    std::vector<HardwareResource> injection_ports_;
    Statistics stats_;
    std::map<std::uint32_t, std::vector<std::size_t>> input_by_timestep_;

    void prepare_input_timesteps();
    void inject_timestep(std::uint32_t timestep, SimTime hw_start_time);
    void process_data(Spike& spike);
    void process_bias(Spike& spike);
    void push_firings(std::size_t layer, std::uint32_t timestep,
                      const std::vector<CoreFiringResult>& firings);
    const std::vector<float>& final_scores() const;
};

}  // namespace soma

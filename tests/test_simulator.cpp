#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/core.hpp"
#include "soma/hw/noc/router.hpp"
#include "soma/hw/soma.hpp"
#include "soma/input_encoder.hpp"
#include "soma/runtime/spatial_template.hpp"
#include "soma/sim/spike_queue.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("test failed: " + message);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("test expects repository root");
        const std::string root = argv[1];
        require(soma::parse_time_ps("6.5 ns") == 6500, "time conversion");

        soma::SpikeQueue queue;
        soma::Spike first;
        first.generated_time = 10;
        first.spike_id = 1;
        soma::Spike second;
        second.generated_time = 10;
        second.spike_id = 2;
        soma::Spike early;
        early.generated_time = 5;
        early.spike_id = 3;
        queue.push(first);
        queue.push(second);
        queue.push(early);
        require(queue.pop().spike_id == 3, "queue time order");
        require(queue.pop().spike_id == 1, "queue stable sequence order");
        require(queue.pop().spike_id == 2, "queue stable sequence tail");

        soma::SpikeQueue synchronized_queue(true);
        soma::Spike later_timestep;
        later_timestep.timestep = 2;
        later_timestep.generated_time = 0;
        later_timestep.spike_id = 4;
        soma::Spike first_timestep;
        first_timestep.timestep = 1;
        first_timestep.generated_time = 100;
        first_timestep.spike_id = 5;
        synchronized_queue.push(later_timestep);
        synchronized_queue.push(first_timestep);
        require(synchronized_queue.pop().spike_id == 5,
                "synchronized queue orders timestep before hardware timestamp");

        const auto hardware = soma::HardwareConfig::load(root + "/arch/hardware.yaml");
        require(hardware.timestep_synchronization(), "execution mode config");
        require(hardware.noc.east_link_hw_latency == 4100,
                "NoC hardware latency schema");
        require(hardware.core.axon_in_hw_latency == 16000,
                "Core hardware latency schema");
        const auto mapping = soma::MappingConfig::load(root + "/compiler/mapping_output/mapping.yaml");
        mapping.validate(static_cast<std::uint32_t>(hardware.noc.router_count()));
        soma::RouterResourceTable routers(hardware);
        const auto& route = mapping.route("input", "conv0");
        const auto a = routers.traverse(0, route);
        const auto b = routers.traverse(0, route);
        require(a.hops == 1, "one-hop static route");
        require(b.hw_congestion_latency > 0, "router output hardware latency contention");

        const auto input = soma::load_input_spikes_csv(root + "/input/input_spike.csv");
        require(input.last_timestep == 2, "input logical timestep range");
        for (const auto& spike : input.spikes) {
            require(spike.timestep >= 1 && spike.generated_time == 0,
                    "synchronized input starts at timestep 1 with zero CSV timestamp");
        }

        soma::SpatialTemplate spatial;
        spatial.cin = 1;
        spatial.cout = 2;
        spatial.weight = {2.0F, 3.0F};
        spatial.plan_pattern_id = {0};
        spatial.plan_dst_base = {0};
        spatial.pattern_ptr = {0, 1};
        spatial.pattern_dst_offset = {0};
        spatial.pattern_weight_offset = {0};
        spatial.validate(1, 2);
        std::vector<float> destination(2, 0.0F);
        const auto updates = spatial.for_each_destination(0, 0.5F,
            [&](std::size_t index, float value) { destination[index] += value; });
        require(updates == 2 && std::abs(destination[0] - 1.0F) < 1e-6F &&
                    std::abs(destination[1] - 1.5F) < 1e-6F,
                "source-major spatial template");

        soma::SomaState soma_state(3, 1.0F, 1.0F, "soft", false);
        soma_state.accumulate(2, 1.5F, 1);
        soma_state.accumulate(0, 2.5F, 1);
        const auto fired = soma_state.fire_all_ordered();
        require(fired.size() == 3 && fired[0].neuron == 0 && fired[1].neuron == 0 &&
                    fired[2].neuron == 2,
                "firing is ordered by neuron id and preserves soft reset");
        require(soma_state.fire_all_ordered().empty(),
                "direct firing drains only the current update candidates");

        soma::LayerMapping direct_mapping;
        direct_mapping.id = "direct_fire";
        direct_mapping.op = soma::LayerOp::Linear;
        direct_mapping.neurons = 3;
        direct_mapping.source_neurons = 1;
        direct_mapping.output_channels = 3;
        direct_mapping.threshold = 1.0F;
        direct_mapping.reset = "soft";
        soma::LayerWeights direct_weights;
        direct_weights.op = soma::LayerOp::Linear;
        direct_weights.dense_weight = {2.5F, 0.0F, 1.5F};
        soma::HardwareConfig direct_hardware;
        direct_hardware.core.synapse_hw_latency = 5;
        direct_hardware.core.soma_update_hw_latency = 10;
        direct_hardware.core.soma_fire_hw_latency = 7;
        soma::Core direct_core(direct_mapping, direct_hardware, direct_weights);
        const auto direct = direct_core.receive(0, 1.0F, 1, 100);
        require(direct.hw_update_finish_time == 135 && direct.hw_finish_time == 156 &&
                    direct.hw_compute_latency == 56,
                "direct firing occupies configured soma fire latency");
        require(direct.firings.size() == 3 && direct.firings[0].fired.neuron == 0 &&
                    direct.firings[1].fired.neuron == 0 && direct.firings[2].fired.neuron == 2 &&
                    direct.firings[0].hw_finish_time == 142 &&
                    direct.firings[1].hw_finish_time == 149 &&
                    direct.firings[2].hw_finish_time == 156,
                "Core returns ordered Data firing times");
        std::cout << "focused tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

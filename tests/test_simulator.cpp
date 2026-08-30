#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
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

        soma::SomaState soma_state(2, 1.0F, 1.0F, "soft", false);
        soma_state.accumulate(0, 2.5F, 0);
        require(soma_state.has_pending(), "threshold candidate is queued");
        require(soma_state.fire_one().has_value() && soma_state.has_pending(),
                "soft-reset fake-spike drain keeps remaining threshold");
        require(soma_state.fire_one().has_value() && !soma_state.has_pending(),
                "fake-spike drain emits one spike at a time");
        std::cout << "focused tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

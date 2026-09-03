#include "soma/common/types.hpp"
#include "soma/config/hardware_config.hpp"
#include "soma/config/mapping_config.hpp"
#include "soma/hw/core.hpp"
#include "soma/hw/noc/router.hpp"
#include "soma/hw/soma.hpp"
#include "soma/hw/synapse.hpp"
#include "soma/hw/tile.hpp"
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
        require(hardware.core.input_fifo && hardware.core.fifo_per_core &&
                    hardware.core.fifo_num_per_core == 1 &&
                    hardware.core.fifo_depth_per_core == 16,
                "per-Core input FIFO schema");
        require(hardware.core.spatial_synapse_hw_latency == 3100 &&
                    hardware.core.dense_synapse_hw_latency == 3800,
                "spatial and dense synapse latency schema");
        require(hardware.core.soma_access_hw_latency == 6000 &&
                    hardware.core.soma_update_hw_latency == 3700,
                "soma access and update latency schema");
        require(hardware.core.threshold_compare_mode == "signed",
                "existing hardware keeps signed threshold comparison by default");
        require(hardware.noc.synchronization_hw_latency(70) == 1'800'000,
                "timestep synchronization latency table");
        const auto truenorth_hardware = soma::HardwareConfig::load(
            root + "/arch/truenorth.yaml");
        require(truenorth_hardware.core.max_neurons == 256 &&
                    truenorth_hardware.core.max_axons == 256 &&
                    truenorth_hardware.core.crossbar_weight_mode == "nonzero_binary" &&
                    truenorth_hardware.core.threshold_compare_mode == "unsigned_promotion" &&
                    truenorth_hardware.core.crossbar_packet_activates_all_neurons &&
                    truenorth_hardware.core.process_inactive_neurons_on_crossbar_event,
                "TrueNorth crossbar behavior is selected by hardware config");
        const auto mapping = soma::MappingConfig::load(root + "/compiler/mapping_output/mapping.yaml");
        mapping.validate(static_cast<std::uint32_t>(hardware.noc.router_count()));
        const auto vgg_mapping = soma::MappingConfig::load(
            root + "/compiler/mapping_output/vgg16_mapping.yaml");
        std::uint64_t vgg_physical_cores = 0;
        for (const auto& layer : vgg_mapping.layers) {
            require(layer.op != soma::LayerOp::AvgPool2d,
                    "VGG Pool is fused instead of mapped as a firing layer");
            vgg_physical_cores +=
                (layer.neurons + hardware.core.max_neurons - 1) /
                hardware.core.max_neurons;
        }
        require(vgg_physical_cores == 279 && vgg_mapping.layer("conv2").pe == 33 &&
                    vgg_mapping.layer("conv2").core == 2 &&
                    vgg_mapping.layer("fc1").source_neurons == 2048,
                "fused VGG graph uses the SANA-FE 279-Core placement");
        const std::vector<std::string> expected_vgg_layers = {
            "input", "conv0", "conv1", "conv2", "conv3", "conv4", "conv5",
            "conv6", "conv7", "conv8", "conv9", "conv10", "conv11", "conv12",
            "fc1", "fc2", "readout"};
        const std::vector<std::uint32_t> expected_vgg_core_starts = {
            0, 6, 70, 134, 166, 198, 214, 230, 246, 254, 262, 270, 272, 274,
            276, 277, 278};
        for (std::size_t index = 0; index < expected_vgg_layers.size(); ++index) {
            const auto& layer = vgg_mapping.layer(expected_vgg_layers[index]);
            const auto global_core = layer.pe * hardware.core.cores_per_pe + layer.core;
            const auto expected_end = index + 1 < expected_vgg_core_starts.size()
                                          ? expected_vgg_core_starts[index + 1]
                                          : 279U;
            require(global_core == expected_vgg_core_starts[index] &&
                        layer.aggregate_core_count == expected_end - global_core,
                    "every fused VGG layer preserves the reference physical-Core range");
        }
        soma::RouterResourceTable routers(hardware);
        const auto& route = mapping.route("input", "conv0");
        const auto a = routers.traverse(0, route);
        const auto b = routers.traverse(0, route);
        require(a.hops == 1, "one-hop static route");
        require(b.hw_congestion_latency > 0, "router output hardware latency contention");
        soma::TileLayout tile_layout(hardware);
        const auto physical_core = tile_layout.core_address(278);
        require(physical_core.tile == 69 && physical_core.core_within_tile == 2 &&
                    physical_core.router == 69,
                "physical Core maps to its Tile/Router");
        const auto physical_route = routers.traverse(0, 0, 69);
        require(physical_route.hops == 69,
                "physical Tile endpoints produce a multi-hop XY route");
        soma::RouterResourceTable fifo_routers(hardware);
        fifo_routers.traverse(0, 0, 1, 0);
        fifo_routers.record_destination_processing(100'000, 10'000);
        fifo_routers.traverse(0, 0, 1, 1);
        // Core 1 可以早于 Core 0 释放；两者不应被误建模成同一个 FIFO。
        fifo_routers.record_destination_processing(50'000, 10'000);

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

        soma::LayerWeights crossbar_weights;
        crossbar_weights.crossbar_axons = 1;
        crossbar_weights.crossbar_words_per_axon = 1;
        crossbar_weights.crossbar_axon_type = {1};
        crossbar_weights.crossbar_rows = {0b11};
        crossbar_weights.crossbar_neuron_weight = {
            0, -3, 0, 0,
            0, 0, 0, 0,
        };
        std::vector<float> signed_crossbar(2, 0.0F);
        const auto signed_updates = soma::SynapseEngine::apply_crossbar(
            crossbar_weights, 1.0F, 0, 0, 0, false,
            [&](std::uint64_t neuron, float value) { signed_crossbar[neuron] += value; });
        std::vector<float> binary_crossbar(2, 0.0F);
        const auto binary_updates = soma::SynapseEngine::apply_crossbar(
            crossbar_weights, 1.0F, 0, 0, 0, true,
            [&](std::uint64_t neuron, float value) { binary_crossbar[neuron] += value; });
        require(signed_updates == 2 && binary_updates == 2 &&
                    signed_crossbar == std::vector<float>({-3.0F, 0.0F}) &&
                    binary_crossbar == std::vector<float>({1.0F, 0.0F}),
                "crossbar signed/nonzero-binary accumulation follows hardware config");

        soma::SomaState soma_state(3, 1.0F, 1.0F, "soft", false);
        const auto first_fire = soma_state.process_neuron(0, 2.5F, true, 0.0F);
        require(first_fire.updated && first_fire.fired && first_fire.fired->neuron == 0 &&
                    std::abs(soma_state.voltage()[0] - 1.5F) < 1e-6F,
                "one timestep emits at most one spike and preserves soft-reset voltage");
        const auto second_fire = soma_state.process_neuron(0, 0.0F, false, 0.0F);
        require(second_fire.updated && second_fire.fired &&
                    std::abs(soma_state.voltage()[0] - 0.5F) < 1e-6F,
                "remaining membrane can fire in the next timestep");
        const auto bias_fire = soma_state.process_neuron(2, 0.0F, false, 1.5F);
        require(bias_fire.updated && bias_fire.fired,
                "bias participates directly in neuron processing");
        soma::SomaState zero_threshold_slot(
            1, 1.0F, 1.0F, "soft", false, 0.0F, "greater_equal", "signed", {0.0F});
        require(zero_threshold_slot.process_neuron(0, 0.0F, true, 0.0F).fired.has_value(),
                "zero-initialized inactive crossbar slot can reproduce event-model firing");
        soma::SomaState signed_threshold_slot(
            1, 177.0F, 1.0F, "soft", false, 0.0F, "greater_equal", "signed");
        require(!signed_threshold_slot.process_neuron(0, -135.0F, true, 0.0F).fired.has_value(),
                "signed threshold comparison keeps negative membrane below positive threshold");
        const std::vector<std::pair<float, float>> nemo_target_states = {
            {9.0F, -312.0F}, {5.0F, -316.0F}, {1.0F, -320.0F}};
        for (const auto& [synaptic_input, expected_reset_voltage] : nemo_target_states) {
            soma::SomaState nemo_threshold_slot(
                1, 177.0F, 1.0F, "soft", false, 0.0F, "greater_equal",
                "unsigned_promotion");
            const auto nemo_threshold_fire =
                nemo_threshold_slot.process_neuron(0, synaptic_input, true, -144.0F);
            require(nemo_threshold_fire.fired.has_value() &&
                        std::abs(nemo_threshold_slot.voltage()[0] - expected_reset_voltage) <
                            1e-6F,
                    "unsigned promotion reproduces each NeMo target firing and linear reset");
        }

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
        direct_hardware.core.dense_synapse_hw_latency = 5;
        direct_hardware.core.spatial_synapse_hw_latency = 7;
        direct_hardware.core.soma_access_hw_latency = 6;
        direct_hardware.core.soma_update_hw_latency = 4;
        direct_hardware.core.soma_fire_hw_latency = 7;
        soma::Core direct_core(direct_mapping, direct_hardware, direct_weights,
                               soma::PhysicalCoreAddress{}, 0, 3);
        const auto accumulation = direct_core.receive(0, 1.0F, 1, 100);
        require(accumulation.hw_finish_time == 115 &&
                    accumulation.synaptic_updates == 3 &&
                    accumulation.hw_synapse_service_latency == 15 &&
                    direct_core.output_scores() == std::vector<float>({0.0F, 0.0F, 0.0F}),
                "packet synapse latency scales with local physical-Core updates");
        const auto direct = direct_core.process_timestep(2, accumulation.hw_finish_time);
        require(direct.mapped_neurons == 3 && direct.updated_neurons == 3 &&
                    direct.hw_finish_time == 159 && direct.hw_compute_latency == 44,
                "neuron processing uses mapped access plus actual update latency");
        require(direct.firings.size() == 2 && direct.firings[0].fired.neuron == 0 &&
                    direct.firings[1].fired.neuron == 2 &&
                    direct.firings[0].hw_finish_time == 132 &&
                    direct.firings[1].hw_finish_time == 159,
                "Core emits at most one ordered firing per neuron");
        const auto next = direct_core.process_timestep(3, direct.hw_finish_time);
        require(next.mapped_neurons == 3 && next.updated_neurons == 2 &&
                    next.firings.size() == 1 && next.firings[0].fired.neuron == 0 &&
                    next.hw_finish_time == 192,
                "remaining membrane is processed in the following timestep");

        soma::LayerMapping partitioned_mapping;
        partitioned_mapping.id = "partitioned";
        partitioned_mapping.op = soma::LayerOp::Conv2d;
        partitioned_mapping.neurons = 2048;
        partitioned_mapping.source_neurons = 1;
        partitioned_mapping.input_channels = 1;
        partitioned_mapping.output_channels = 2;
        partitioned_mapping.physical_neuron_order = "channel_major";
        soma::LayerWeights partitioned_weights;
        partitioned_weights.op = soma::LayerOp::Conv2d;
        partitioned_weights.spatial = spatial;
        std::vector<std::uint32_t> partitions;
        soma::SynapseEngine::for_each_destination_partition(
            partitioned_weights, 0, partitioned_mapping, 1024,
            [&](std::uint32_t partition) { partitions.push_back(partition); });
        require(partitions == std::vector<std::uint32_t>({0, 1}),
                "one source firing packetizes once per destination physical Core");
        std::vector<float> first_partition_updates(2048, 0.0F);
        std::vector<float> second_partition_updates(2048, 0.0F);
        const auto first_partition_count = soma::SynapseEngine::apply_to_physical_range(
            partitioned_weights, 0, 1.0F, partitioned_mapping, 0, 1024,
            [&](std::uint64_t neuron, float value) { first_partition_updates[neuron] += value; });
        const auto second_partition_count = soma::SynapseEngine::apply_to_physical_range(
            partitioned_weights, 0, 1.0F, partitioned_mapping, 1024, 2048,
            [&](std::uint64_t neuron, float value) { second_partition_updates[neuron] += value; });
        require(first_partition_count == 1 && second_partition_count == 1 &&
                    std::abs(first_partition_updates[0] - 2.0F) < 1e-6F &&
                    std::abs(second_partition_updates[1] - 3.0F) < 1e-6F,
                "channel-major packets apply only destination-Core-local synaptic updates");
        soma::Core spatial_core(partitioned_mapping, direct_hardware, partitioned_weights,
                                soma::PhysicalCoreAddress{}, 0, 1024);
        const auto spatial_accumulation = spatial_core.receive(0, 1.0F, 1, 200);
        require(spatial_accumulation.synaptic_updates == 1 &&
                    spatial_accumulation.hw_synapse_service_latency == 7 &&
                    spatial_accumulation.hw_finish_time == 207,
                "spatial Core selects spatial synapse latency");

        soma::LayerMapping full_core_mapping;
        full_core_mapping.id = "full_core";
        full_core_mapping.op = soma::LayerOp::Linear;
        full_core_mapping.neurons = 1024;
        full_core_mapping.source_neurons = 1;
        full_core_mapping.output_channels = 1;
        full_core_mapping.threshold = 1.0F;
        soma::LayerWeights full_core_weights;
        full_core_weights.op = soma::LayerOp::Linear;
        full_core_weights.bias = {0.5F};
        soma::Core full_core(full_core_mapping, hardware, full_core_weights,
                             soma::PhysicalCoreAddress{}, 0, 1024);
        const auto no_traffic = full_core.process_timestep(1, 0);
        require(no_traffic.hw_finish_time +
                    hardware.noc.synchronization_hw_latency(70) == 11'732'800,
                "1024-neuron no-traffic timestep includes soma and synchronization latency");
        std::cout << "focused tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

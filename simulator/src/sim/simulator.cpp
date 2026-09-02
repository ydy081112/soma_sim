#include "soma/sim/simulator.hpp"

#include "soma/hw/synapse.hpp"

#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>

namespace soma {

Simulator::Simulator(SimulatorOptions options)
    : options_(std::move(options)),
      hardware_(HardwareConfig::load(options_.hardware_path)),
      mapping_(MappingConfig::load(options_.mapping_path)),
      weights_(WeightStore::load(options_.weights_path, mapping_)),
      input_(load_input_spikes_csv(options_.input_spikes_path)),
      queue_(hardware_.timestep_synchronization()),
      routers_(hardware_), tile_layout_(hardware_), stats_(mapping_, hardware_) {
    // 所有容量、映射和权重检查都在入队前完成，避免热路径携带恢复逻辑。
    mapping_.validate(static_cast<std::uint32_t>(hardware_.noc.router_count()));
    for (const auto& layer : mapping_.layers) {
        if (layer.pe >= hardware_.core.pe_count) throw std::runtime_error(layer.id + ": PE id 越界");
        if (layer.core >= hardware_.core.cores_per_pe) throw std::runtime_error(layer.id + ": Core id 越界");
    }

    layer_runtime_.resize(mapping_.layers.size());
    axon_out_resources_.resize(tile_layout_.total_cores());
    if (hardware_.core.source_packet_fifo) {
        source_blocking_offsets_.resize(tile_layout_.total_cores(), 0);
        source_pending_packets_.resize(tile_layout_.total_cores());
        source_packet_active_.resize(tile_layout_.total_cores(), 0);
    }
    std::vector<int> physical_core_owner(tile_layout_.total_cores(), -1);
    std::set<std::uint32_t> mapped_tile_ids;
    std::uint64_t physical_core_count = 0;
    for (const auto& layer : mapping_.layers) {
        auto& runtime = layer_runtime_.at(layer.index);
        const auto partition_count = static_cast<std::uint32_t>(
            (layer.neurons + hardware_.core.max_neurons - 1) / hardware_.core.max_neurons);
        if (layer.aggregate_core_count != 0 && layer.aggregate_core_count != partition_count) {
            throw std::runtime_error(layer.id + ": aggregate_core_count 与 max_neurons 分区不一致");
        }
        runtime.first_global_core = layer.pe * hardware_.core.cores_per_pe + layer.core;
        runtime.addresses.reserve(partition_count);
        if (layer.op != LayerOp::Input) runtime.cores.reserve(partition_count);
        for (std::uint32_t partition = 0; partition < partition_count; ++partition) {
            const auto global_core = runtime.first_global_core + partition;
            if (global_core >= physical_core_owner.size()) {
                throw std::runtime_error(layer.id + ": physical Core mapping 超出硬件容量");
            }
            if (physical_core_owner[global_core] >= 0) {
                throw std::runtime_error(layer.id + ": physical Core mapping 与其他 layer 重叠");
            }
            physical_core_owner[global_core] = static_cast<int>(layer.index);
            const auto address = tile_layout_.core_address(global_core);
            runtime.addresses.push_back(address);
            mapped_tile_ids.insert(address.tile);
            if (layer.op != LayerOp::Input) {
                std::uint32_t max_connection_delay = 0;
                for (const auto& connection : mapping_.connections) {
                    if (connection.to == layer.id)
                        max_connection_delay = std::max(max_connection_delay, connection.delay);
                }
                const auto physical_begin = static_cast<std::uint64_t>(partition) *
                                            hardware_.core.max_neurons;
                const auto neuron_count = std::min<std::uint64_t>(
                    hardware_.core.max_neurons, layer.neurons - physical_begin);
                runtime.cores.push_back(std::make_unique<Core>(
                    layer, hardware_, weights_.at(layer.id), address,
                    physical_begin, neuron_count, max_connection_delay));
            }
        }
        physical_core_count += partition_count;
    }
    mapped_tiles_ = mapped_tile_ids.size();
    stats_.set_physical_topology(physical_core_count, mapped_tiles_);
    prepare_input_timesteps();
}

void Simulator::prepare_input_timesteps() {
    for (std::size_t index = 0; index < input_.spikes.size(); ++index) {
        input_by_timestep_[input_.spikes[index].timestep].push_back(index);
    }
}

void Simulator::inject_timestep(std::uint32_t timestep, SimTime hw_start_time) {
    const auto group = input_by_timestep_.find(timestep);
    if (group == input_by_timestep_.end()) return;
    for (const auto index : group->second) {
        const auto& record = input_.spikes[index];
        const auto source = mapping_.index_of(record.layer_id);
        const auto& layer = mapping_.layer(source);
        if (layer.op != LayerOp::Input || !layer.virtual_input) {
            throw std::runtime_error("CSV layer_id 必须指向 virtual input layer: " + record.layer_id);
        }
        if (record.source_neuron >= layer.neurons) throw std::runtime_error("CSV src_neuron 越界");
        // 同步模式不读取 CSV 的绝对时间；本 timestep 从上一批实际完成时刻开始。
        stats_.record_emit(source, record.timestep, hw_start_time);
        enqueue_packets(source, record.source_neuron, record.value, record.timestep,
                        hw_start_time, hw_start_time, record.spike_id);
    }
}

void Simulator::process_neurons(std::uint32_t timestep, SimTime hw_start_time) {
    // 输入 packet 可先以 timestep start 为时标入队；此阶段只生成 firing，不消费 queue。
    // 所有 Core 读取上一 timestep accumulation，并保留各自 neuron loop 中的真实完成时刻。
    std::vector<std::pair<std::size_t, std::vector<CoreFiringResult>>> pending_firings;
    for (const auto& layer : mapping_.layers) {
        if (layer.op == LayerOp::Input) continue;
        const auto layer_host_start = std::chrono::steady_clock::now();
        for (auto& core : layer_runtime_.at(layer.index).cores) {
            auto result = core->process_timestep(timestep, hw_start_time);
            stats_.add_soma_hw_latency(timestep, result.hw_soma_service_latency,
                                       result.hw_compute_latency);
            stats_.add_neuron_energy(result.updated_neurons);
            stats_.record_neuron_processing(layer.index, timestep, result.hw_finish_time);
            pending_firings.emplace_back(layer.index, std::move(result.firings));
        }
        const auto layer_host_end = std::chrono::steady_clock::now();
        stats_.record_host_latency(
            layer.index, timestep,
            std::chrono::duration<double>(layer_host_end - layer_host_start).count());
    }
    for (const auto& [layer, firings] : pending_firings) {
        push_firings(layer, timestep, firings);
    }
}

void Simulator::push_firings(std::size_t layer_index, std::uint32_t timestep,
                             const std::vector<CoreFiringResult>& firings) {
    for (const auto& firing : firings) {
        stats_.record_emit(layer_index, timestep, firing.hw_finish_time);
        stats_.add_fire_energy();
        if (mapping_.outgoing(layer_index).empty()) continue;
        enqueue_packets(layer_index, firing.fired.neuron, firing.fired.value, timestep,
                        firing.hw_finish_time, firing.hw_finish_time);
    }
}

PhysicalCoreAddress Simulator::source_core_address(std::size_t layer_index,
                                                   std::uint64_t logical_neuron) const {
    const auto& layer = mapping_.layer(layer_index);
    const auto physical = layer.physical_neuron_index(logical_neuron);
    const auto partition = static_cast<std::size_t>(physical / hardware_.core.max_neurons);
    return layer_runtime_.at(layer_index).addresses.at(partition);
}

void Simulator::enqueue_packets(std::size_t source_layer, std::uint64_t source_neuron,
                                float value, std::uint32_t timestep,
                                SimTime generated_time, SimTime current_time,
                                std::uint64_t spike_id) {
    const auto source_address = source_core_address(source_layer, source_neuron);
    for (const auto connection_index : mapping_.outgoing(source_layer)) {
        const auto& connection = mapping_.connections.at(connection_index);
        const auto target_index = mapping_.index_of(connection.to);
        const auto& target = mapping_.layer(target_index);
        const auto& connection_weights = weights_.connection(connection_index);
        SynapseEngine::for_each_destination_partition(
            connection_weights, source_neuron, target, hardware_.core.max_neurons,
            [&](std::uint32_t destination_partition) {
            Spike packet;
            if (hardware_.core.source_packet_fifo) {
                const auto injection = axon_out_resources_.at(source_address.global_core).reserve(
                    current_time, hardware_.core.axon_out_hw_latency);
                stats_.add_inject_hw_latency(
                    timestep, injection.hw_finish_time - current_time);
                packet.generated_time = injection.hw_finish_time;
                packet.current_time = injection.hw_finish_time;
                packet.unblocked_send_time = injection.hw_finish_time;
            } else {
                packet.generated_time = generated_time;
                packet.current_time = current_time;
            }
            packet.spike_id = spike_id;
            packet.timestep = timestep;
            packet.source_layer = source_layer;
            packet.source_neuron = source_neuron;
            packet.source_physical_core = source_address.global_core;
            packet.destination_layer = target_index;
            packet.connection = connection_index;
            packet.destination_partition = destination_partition;
            packet.value = value;
            if (hardware_.core.source_packet_fifo) stage_source_packet(std::move(packet));
            else queue_.push(std::move(packet));
        });
    }
}

void Simulator::stage_source_packet(Spike packet) {
    const auto core = packet.source_physical_core;
    if (source_packet_active_.at(core) == 0) {
        packet.generated_time = packet.unblocked_send_time + source_blocking_offsets_.at(core);
        packet.current_time = packet.generated_time;
        source_packet_active_[core] = 1;
        queue_.push(std::move(packet));
    } else {
        source_pending_packets_.at(core).push_back(std::move(packet));
    }
}

void Simulator::release_next_source_packet(std::uint32_t source_core) {
    auto& pending = source_pending_packets_.at(source_core);
    if (pending.empty()) {
        source_packet_active_[source_core] = 0;
        return;
    }
    auto packet = std::move(pending.front());
    pending.pop_front();
    packet.generated_time = packet.unblocked_send_time + source_blocking_offsets_.at(source_core);
    packet.current_time = packet.generated_time;
    queue_.push(std::move(packet));
}

void Simulator::process_data(Spike& spike) {
    // Packet 端点由 mapping 中的 physical Core 放置决定，路径使用确定性 XY routing。
    const auto target_index = spike.destination_layer;
    const auto& target_runtime = layer_runtime_.at(target_index);
    const auto& destination = target_runtime.addresses.at(spike.destination_partition);
    const auto source = tile_layout_.core_address(spike.source_physical_core);

    if (!hardware_.core.source_packet_fifo) {
        // 既有路径：packet 出队时占用 source Core 的 axon-out。
        const auto injection = axon_out_resources_.at(source.global_core).reserve(
            spike.current_time, hardware_.core.axon_out_hw_latency);
        stats_.add_inject_hw_latency(
            spike.timestep, injection.hw_finish_time - spike.current_time);
        spike.current_time = injection.hw_finish_time;
    }

    // noc
    const auto noc = hardware_.core.source_packet_fifo
        ? routers_.traverse(spike.current_time, source.router,
                            source.core_within_tile, destination.router,
                            destination.core_within_tile)
        : routers_.traverse(spike.current_time, source.router,
                            destination.router, destination.core_within_tile);
    if (hardware_.core.source_packet_fifo) {
        source_blocking_offsets_.at(source.global_core) += noc.hw_source_blocking_latency;
    }
    stats_.add_noc_hw_latency(spike.timestep, noc);
    spike.current_time = noc.hw_arrival_time;
    
    // core
    auto& core = *target_runtime.cores.at(spike.destination_partition);
    const auto& connection = mapping_.connections.at(spike.connection);
    const auto receive = core.receive(spike.source_neuron, spike.value, spike.timestep,
                                      spike.current_time, &weights_.connection(spike.connection),
                                      connection.delay);
    spike.current_time = receive.hw_finish_time;
    const auto receive_service = receive.hw_axon_in_service_latency +
                                 receive.hw_synapse_service_latency;
    routers_.record_destination_processing(
        receive.hw_finish_time - receive_service, receive_service);

    stats_.add_synapse_hw_latency(spike.timestep, receive.hw_synapse_service_latency,
                                  receive.hw_compute_latency);
    stats_.add_data_energy(noc, receive.synaptic_updates, connection.hardware_type);
    stats_.record_packet(target_index, spike.timestep, receive.synaptic_updates,
                         noc, receive.hw_finish_time);
    if (hardware_.core.source_packet_fifo) {
        release_next_source_packet(source.global_core);
    }
}

std::vector<float> Simulator::final_scores() const {
    // 优先选择显式 readout，否则使用 mapping 中最后一个无后继的计算层。
    for (std::size_t i = mapping_.layers.size(); i > 0; --i) {
        const auto& layer = mapping_.layers[i - 1];
        if (layer.op != LayerOp::Input &&
            (layer.readout || mapping_.outgoing(i - 1).empty())) {
            std::vector<float> scores(static_cast<std::size_t>(layer.neurons), 0.0F);
            for (const auto& core : layer_runtime_.at(i - 1).cores) {
                const auto& local_scores = core->output_scores();
                for (std::size_t local = 0; local < local_scores.size(); ++local) {
                    const auto physical = core->physical_neuron_begin() + local;
                    const auto logical = layer.logical_neuron_index(physical);
                    scores.at(static_cast<std::size_t>(logical)) = local_scores[local];
                }
            }
            return scores;
        }
    }
    throw std::runtime_error("mapping 缺少 output/readout layer");
}

SimulationResult Simulator::run() {
    if (!hardware_.timestep_synchronization()) {
        throw std::logic_error("当前 HardwareConfig 不使用 timestep synchronization");
    }
    return run_timestep_synchronization();
}

SimulationResult Simulator::run_timestep_synchronization() {
    const auto host_start = std::chrono::steady_clock::now();
    std::uint64_t events = 0;
    bool completed = true;
    SimTime next_timestep_hw_start = 0;
    for (std::uint32_t timestep = 1; timestep <= input_.last_timestep; ++timestep) {
        const auto timestep_host_start = std::chrono::steady_clock::now();
        if (options_.max_events != 0 && events >= options_.max_events) {
            completed = false;
            break;
        }
        stats_.begin_timestep(timestep, next_timestep_hw_start);
        if (hardware_.core.source_packet_fifo) {
            std::fill(source_blocking_offsets_.begin(), source_blocking_offsets_.end(), 0);
            if (std::any_of(source_packet_active_.begin(), source_packet_active_.end(),
                            [](std::uint8_t active) { return active != 0; })) {
                throw std::logic_error("timestep 开始前 source packet FIFO 必须为空");
            }
        }
        if (!queue_.empty()) {
            throw std::logic_error("timestep 开始前 global queue 必须为空");
        }
        // input 从 timestep start 进入 queue；内部 firing 随后按各 Core 的真实时标交错入队。
        inject_timestep(timestep, next_timestep_hw_start);
        process_neurons(timestep, next_timestep_hw_start);

        // 当前 timestep 的真实 Data、NoC 和 synaptic accumulation 全部完成后才进入 barrier。
        while (!queue_.empty()) {
            if (options_.max_events != 0 && events >= options_.max_events) {
                completed = false;
                break;
            }
            Spike spike = queue_.pop();
            process_data(spike);
            ++events;
        }
        if (!completed) break;
        const auto synchronization_hw_latency =
            hardware_.noc.synchronization_hw_latency(mapped_tiles_);
        next_timestep_hw_start = stats_.hw_latency() + synchronization_hw_latency;
        stats_.complete_timestep(timestep, next_timestep_hw_start,
                                 synchronization_hw_latency);
        const auto timestep_host_end = std::chrono::steady_clock::now();
        stats_.record_timestep_host_latency(
            timestep,
            std::chrono::duration<double>(timestep_host_end - timestep_host_start).count());
    }
    const auto host_end = std::chrono::steady_clock::now();
    stats_.set_host_latency(std::chrono::duration<double>(host_end - host_start).count());
    stats_.set_stopped_early(!completed);
    const auto scores = final_scores();
    stats_.write(options_.output_dir, scores, input_.expected_output);
    return SimulationResult{stats_.hw_latency(), stats_.host_latency_s(),
                            stats_.processed_spikes(), completed, scores};
}

}  // namespace soma

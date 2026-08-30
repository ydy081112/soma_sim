#include "soma/sim/simulator.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace soma {

Simulator::Simulator(SimulatorOptions options)
    : options_(std::move(options)),
      hardware_(HardwareConfig::load(options_.hardware_path)),
      mapping_(MappingConfig::load(options_.mapping_path)),
      weights_(WeightStore::load(options_.weights_path, mapping_)),
      input_(load_input_spikes_csv(options_.input_spikes_path)),
      routers_(hardware_), stats_(mapping_, hardware_) {
    mapping_.validate(static_cast<std::uint32_t>(hardware_.noc.router_count()));
    std::uint32_t max_pe = 0;
    for (const auto& layer : mapping_.layers) {
        if (layer.pe >= hardware_.core.pe_count) throw std::runtime_error(layer.id + ": PE id 越界");
        if (layer.core >= hardware_.core.cores_per_pe) throw std::runtime_error(layer.id + ": Core id 越界");
        if (layer.partition != "aggregated" && layer.op != LayerOp::Input &&
            layer.neurons > hardware_.core.max_neurons) {
            throw std::runtime_error(layer.id + ": partition neuron 数超过 Core 容量");
        }
        max_pe = std::max(max_pe, layer.pe);
    }
    injection_ports_.resize(static_cast<std::size_t>(max_pe) + 1);
    cores_.resize(mapping_.layers.size());
    for (const auto& layer : mapping_.layers) {
        if (layer.op != LayerOp::Input) {
            cores_[layer.index] = std::make_unique<Core>(layer, hardware_, weights_.at(layer.id));
        }
    }
    load_input_queue();
}

void Simulator::load_input_queue() {
    // 每个逻辑 timestep 只建立一组轻量 Bias 事件；bias 更新仍由事件队列调度，
    // 不引入逐 ps/tick 的 time-driven 扫描。
    std::vector<std::pair<SimTime, std::uint32_t>> steps;
    for (const auto& record : input_.spikes) {
        const auto item = std::make_pair(record.generated_time, record.timestep);
        if (steps.empty() || steps.back() != item) steps.push_back(item);
    }
    for (const auto& step : steps) {
        for (const auto& layer : mapping_.layers) {
            if (layer.op == LayerOp::Input || !cores_.at(layer.index)->has_bias()) continue;
            Spike bias;
            bias.kind = SpikeKind::Bias;
            bias.generated_time = step.first;
            bias.current_time = step.first;
            bias.timestep = step.second;
            bias.source_layer = layer.index;
            queue_.push(std::move(bias));
        }
    }
    for (const auto& record : input_.spikes) {
        const auto source = mapping_.index_of(record.layer_id);
        const auto& layer = mapping_.layer(source);
        if (layer.op != LayerOp::Input || !layer.virtual_input) {
            throw std::runtime_error("CSV layer_id 必须指向 virtual input layer: " + record.layer_id);
        }
        if (record.source_neuron >= layer.neurons) throw std::runtime_error("CSV src_neuron 越界");
        Spike spike;
        spike.kind = SpikeKind::Data;
        spike.generated_time = record.generated_time;
        spike.current_time = record.generated_time;
        spike.spike_id = record.spike_id;
        spike.timestep = record.timestep;
        spike.source_layer = source;
        spike.source_neuron = record.source_neuron;
        spike.value = record.value;
        queue_.push(std::move(spike));
        stats_.record_emit(source, record.timestep, record.generated_time);
    }
}

void Simulator::process_bias(Spike& spike) {
    auto& core = *cores_.at(spike.source_layer);
    const auto result = core.apply_bias(spike.timestep, spike.current_time);
    spike.current_time = result.hw_finish_time;
    stats_.add_compute_hw_latency(result.hw_compute_latency);
    stats_.add_bias_energy(result.synaptic_updates);
    stats_.record_state_updates(spike.source_layer, spike.timestep, result.synaptic_updates,
                                spike.current_time);
    schedule_drain(spike.source_layer, spike.timestep, spike.current_time);
}

void Simulator::schedule_drain(std::size_t layer, std::uint32_t timestep, SimTime hw_time) {
    auto& core = *cores_.at(layer);
    if (!core.has_pending_fire() || core.drain_scheduled()) return;
    core.set_drain_scheduled(true);
    Spike fake;
    fake.kind = SpikeKind::SomaDrain;
    fake.generated_time = hw_time;
    fake.current_time = hw_time;
    fake.timestep = timestep;
    fake.source_layer = layer;
    queue_.push(std::move(fake));
}

void Simulator::process_data(Spike& spike) {
    const auto& source = mapping_.layer(spike.source_layer);
    if (source.next.empty()) return;
    const auto target_index = mapping_.index_of(source.next);
    const auto& target = mapping_.layer(target_index);
    const auto& route = mapping_.route(source.id, target.id);

    const auto injection = injection_ports_.at(source.pe).reserve(
        spike.current_time, hardware_.core.axon_out_hw_latency);
    stats_.add_inject_hw_latency(injection.hw_finish_time - spike.current_time);
    spike.current_time = injection.hw_finish_time;

    const auto noc = routers_.traverse(spike.current_time, route);
    stats_.add_noc_hw_latency(noc);
    spike.current_time = noc.hw_arrival_time;

    auto& core = *cores_.at(target_index);
    const auto receive = core.receive(spike.source_neuron, spike.value, spike.timestep, spike.current_time);
    spike.current_time = receive.hw_finish_time;
    stats_.add_compute_hw_latency(receive.hw_compute_latency);
    stats_.add_data_energy(noc, receive.synaptic_updates);
    stats_.record_data(target_index, spike.timestep, receive.synaptic_updates, spike.current_time);
    schedule_drain(target_index, spike.timestep, spike.current_time);
}

void Simulator::process_drain(Spike& spike) {
    auto& core = *cores_.at(spike.source_layer);
    core.set_drain_scheduled(false);
    const auto result = core.drain_one(spike.current_time);
    spike.current_time = result.hw_finish_time;
    stats_.add_compute_hw_latency(result.hw_compute_latency);
    if (result.fired) {
        stats_.record_emit(spike.source_layer, spike.timestep, spike.current_time);
        stats_.add_fire_energy();
        const auto& layer = mapping_.layer(spike.source_layer);
        if (!layer.next.empty()) {
            Spike generated;
            generated.kind = SpikeKind::Data;
            generated.generated_time = spike.current_time;
            generated.current_time = spike.current_time;
            generated.spike_id = spike.sequence_id;
            generated.timestep = spike.timestep;
            generated.source_layer = spike.source_layer;
            generated.source_neuron = result.fired->neuron;
            generated.value = result.fired->value;
            queue_.push(std::move(generated));
        }
    }
    schedule_drain(spike.source_layer, spike.timestep, spike.current_time);
}

const std::vector<float>& Simulator::final_scores() const {
    for (std::size_t i = mapping_.layers.size(); i > 0; --i) {
        const auto& layer = mapping_.layers[i - 1];
        if (layer.op != LayerOp::Input && (layer.readout || layer.next.empty())) {
            return cores_.at(i - 1)->output_scores();
        }
    }
    throw std::runtime_error("mapping 缺少 output/readout layer");
}

SimulationResult Simulator::run() {
    const auto host_start = std::chrono::steady_clock::now();
    std::uint64_t events = 0;
    bool completed = true;
    while (!queue_.empty()) {
        if (options_.max_events != 0 && events >= options_.max_events) {
            completed = false;
            break;
        }
        Spike spike = queue_.pop();
        const std::size_t metric_layer = spike.kind == SpikeKind::Data &&
                                                 !mapping_.layer(spike.source_layer).next.empty()
                                             ? mapping_.index_of(mapping_.layer(spike.source_layer).next)
                                             : spike.source_layer;
        const auto host_event_start = std::chrono::steady_clock::now();
        if (spike.kind == SpikeKind::Data) process_data(spike);
        else if (spike.kind == SpikeKind::Bias) process_bias(spike);
        else process_drain(spike);
        const auto host_event_end = std::chrono::steady_clock::now();
        stats_.record_host_latency(
            metric_layer, spike.timestep,
            std::chrono::duration<double>(host_event_end - host_event_start).count());
        ++events;
    }
    const auto host_end = std::chrono::steady_clock::now();
    stats_.set_host_latency(std::chrono::duration<double>(host_end - host_start).count());
    stats_.set_stopped_early(!completed);
    const auto& scores = final_scores();
    stats_.write(options_.output_dir, scores, input_.expected_output);
    return SimulationResult{stats_.hw_latency(), stats_.host_latency_s(),
                            stats_.processed_spikes(), completed, scores};
}

}  // namespace soma

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
      queue_(hardware_.timestep_synchronization()),
      routers_(hardware_), stats_(mapping_, hardware_) {
    // 所有容量、映射和权重检查都在入队前完成，避免热路径携带恢复逻辑。
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
        Spike spike;
        // 同步模式不读取 CSV 的绝对时间；本 timestep 从上一批实际完成时刻开始。
        spike.generated_time = hw_start_time;
        spike.current_time = hw_start_time;
        spike.spike_id = record.spike_id;
        spike.timestep = record.timestep;
        spike.source_layer = source;
        spike.source_neuron = record.source_neuron;
        spike.value = record.value;
        queue_.push(std::move(spike));
        stats_.record_emit(source, record.timestep, hw_start_time);
    }
}

SimTime Simulator::process_neurons(std::uint32_t timestep, SimTime hw_start_time) {
    if (!queue_.empty()) {
        throw std::logic_error("neuron processing 开始前 global queue 必须为空");
    }
    // 所有 Core 先读取上一 timestep 的 accumulation；代码顺序不改变各 Core 的独立资源并行性。
    SimTime data_phase_start = hw_start_time;
    std::vector<std::pair<std::size_t, std::vector<CoreFiringResult>>> pending_firings;
    for (const auto& layer : mapping_.layers) {
        if (layer.op == LayerOp::Input) continue;
        const auto host_start = std::chrono::steady_clock::now();
        auto result = cores_.at(layer.index)->process_timestep(timestep, hw_start_time);
        const auto host_end = std::chrono::steady_clock::now();
        stats_.add_compute_hw_latency(result.hw_compute_latency);
        stats_.add_neuron_energy(result.updated_neurons);
        stats_.record_neuron_processing(layer.index, timestep, result.hw_finish_time);
        data_phase_start = std::max(data_phase_start, result.hw_finish_time);
        pending_firings.emplace_back(layer.index, std::move(result.firings));
        stats_.record_host_latency(
            layer.index, timestep,
            std::chrono::duration<double>(host_end - host_start).count());
    }
    for (const auto& [layer, firings] : pending_firings) {
        push_firings(layer, timestep, firings, data_phase_start);
    }
    return data_phase_start;
}

void Simulator::push_firings(std::size_t layer_index, std::uint32_t timestep,
                             const std::vector<CoreFiringResult>& firings,
                             SimTime data_phase_start) {
    const auto& layer = mapping_.layer(layer_index);
    for (const auto& firing : firings) {
        stats_.record_emit(layer_index, timestep, firing.hw_finish_time);
        stats_.add_fire_energy();
        if (layer.next.empty()) continue;
        Spike generated;
        generated.generated_time = firing.hw_finish_time;
        generated.current_time = std::max(firing.hw_finish_time, data_phase_start);
        generated.timestep = timestep;
        generated.source_layer = layer_index;
        generated.source_neuron = firing.fired.neuron;
        generated.value = firing.fired.value;
        queue_.push(std::move(generated));
    }
}

void Simulator::process_data(Spike& spike) {
    // route 的唯一来源是 mapping；data spike 依次经过注入、NoC 和目标 Core。
    // 根据 spike 携带的 source_layer，去 mapping 里面找到它当前所在 layer 的硬件映射信息
    const auto& source = mapping_.layer(spike.source_layer);
    if (source.next.empty()) return;
    // 找到下一层在 mapping_.layers 数组中的 index
    const auto target_index = mapping_.index_of(source.next);
    // 现在根据刚才找到的 index，把下一层完整 mapping 信息取出来
    const auto& target = mapping_.layer(target_index);
    // 根据 source layer 和 target layer，去 mapping.yaml 里找到编译器提前算好的静态 NoC route
    const auto& route = mapping_.route(source.id, target.id);

    // spike 在进入 NoC 之前，要先经过 source PE 的 axon output / injection port
    // 语法: 这里选中了 source.pe 这个PE的 injection port 部件（对象）
    // 例: injection_ports_[0]  → PE0 的 injection port
    const auto injection = injection_ports_.at(source.pe).reserve(
        spike.current_time,
        hardware_.core.axon_out_hw_latency);// 这个 spike 使用 axon output 资源本身需要多长时间
    stats_.add_inject_hw_latency(injection.hw_finish_time - spike.current_time);
    spike.current_time = injection.hw_finish_time;

    // noc
    const auto noc = routers_.traverse(spike.current_time, route);
    stats_.add_noc_hw_latency(noc);
    spike.current_time = noc.hw_arrival_time;
    
    // core
    auto& core = *cores_.at(target_index);
    const auto receive = core.receive(spike.source_neuron, spike.value, spike.timestep, spike.current_time);
    spike.current_time = receive.hw_finish_time;

    stats_.add_compute_hw_latency(receive.hw_compute_latency);
    stats_.add_data_energy(noc, receive.synaptic_updates);
    stats_.record_data(target_index, spike.timestep, receive.synaptic_updates,
                       receive.hw_finish_time);
}

const std::vector<float>& Simulator::final_scores() const {
    // 优先选择显式 readout，否则使用 mapping 中最后一个无后继的计算层。
    for (std::size_t i = mapping_.layers.size(); i > 0; --i) {
        const auto& layer = mapping_.layers[i - 1];
        if (layer.op != LayerOp::Input && (layer.readout || layer.next.empty())) {
            return cores_.at(i - 1)->output_scores();
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
        if (options_.max_events != 0 && events >= options_.max_events) {
            completed = false;
            break;
        }
        // t 的 neuron phase 只消费 t-1 buffer；随后本步全部 Data 只写下一批 buffer。
        const auto data_phase_start = process_neurons(timestep, next_timestep_hw_start);
        inject_timestep(timestep, data_phase_start);

        // 当前 timestep 的真实 Data、NoC 和 synaptic accumulation 全部完成后才进入 barrier。
        while (!queue_.empty()) {
            if (options_.max_events != 0 && events >= options_.max_events) {
                completed = false;
                break;
            }
            Spike spike = queue_.pop();
            // global queue 只包含 Data，host 开销归到其目标层。
            const std::size_t metric_layer = !mapping_.layer(spike.source_layer).next.empty()
                                                 ? mapping_.index_of(mapping_.layer(spike.source_layer).next)
                                                 : spike.source_layer;
            const auto host_event_start = std::chrono::steady_clock::now();
            process_data(spike);
            const auto host_event_end = std::chrono::steady_clock::now();
            stats_.record_host_latency(
                metric_layer, spike.timestep,
                std::chrono::duration<double>(host_event_end - host_event_start).count());
            ++events;
        }
        if (!completed) break;
        next_timestep_hw_start = stats_.hw_latency();
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

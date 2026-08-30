#include "soma/sim/spike_queue.hpp"

#include <stdexcept>

namespace soma {

SpikeQueue::SpikeQueue(bool timestep_first) : queue_(Later{timestep_first}) {}

std::uint64_t SpikeQueue::push(Spike spike) {
    // sequence_id 在入队时统一分配，为同 generated_time 的事件提供稳定顺序。
    spike.sequence_id = next_sequence_++;
    const auto sequence = spike.sequence_id;
    queue_.push(std::move(spike));
    return sequence;
}

Spike SpikeQueue::pop() {
    if (queue_.empty()) throw std::runtime_error("SpikeQueue 为空");
    Spike spike = queue_.top();
    queue_.pop();
    return spike;
}

}  // namespace soma

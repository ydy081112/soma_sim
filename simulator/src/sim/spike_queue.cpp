#include "soma/sim/spike_queue.hpp"

#include <stdexcept>

namespace soma {

std::uint64_t SpikeQueue::push(Spike spike) {
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


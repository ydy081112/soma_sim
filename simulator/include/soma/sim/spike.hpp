#pragma once

#include "soma/common/types.hpp"

#include <cstdint>

namespace soma {

struct Spike {
    SimTime generated_time = 0;  // 全局队列排序键，不在传输途中修改。
    SimTime current_time = 0;    // 经过每个硬件资源后单调推进。
    SimTime unblocked_send_time = 0;  // source FIFO 回压前的 axon-out 完成时刻。
    std::uint64_t sequence_id = 0;
    std::uint64_t spike_id = 0;
    std::uint32_t timestep = 0;
    std::size_t source_layer = 0;
    std::uint64_t source_neuron = 0;
    std::uint32_t source_physical_core = 0;
    std::size_t destination_layer = 0;
    std::size_t connection = 0;
    std::uint32_t destination_partition = 0;
    std::uint32_t destination_axon = 0;
    float value = 1.0F;
    // streaming input 时标识一条 CSV record 的最后一个 fan-out packet。
    bool input_record_last = false;
};

}  // namespace soma

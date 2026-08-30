#pragma once

#include "soma/sim/spike.hpp"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace soma {

class SpikeQueue {
public:
    explicit SpikeQueue(bool timestep_first = false);
    std::uint64_t push(Spike spike);
    Spike pop();
    bool empty() const { return queue_.empty(); }
    std::size_t size() const { return queue_.size(); }

private:
    struct Later {
        bool timestep_first = false;

        bool operator()(const Spike& lhs, const Spike& rhs) const {
            if (timestep_first && lhs.timestep != rhs.timestep) {
                return lhs.timestep > rhs.timestep;
            }
            if (lhs.generated_time != rhs.generated_time) return lhs.generated_time > rhs.generated_time;
            return lhs.sequence_id > rhs.sequence_id;
        }
    };
    std::priority_queue<Spike, std::vector<Spike>, Later> queue_;
    std::uint64_t next_sequence_ = 0;
};

}  // namespace soma

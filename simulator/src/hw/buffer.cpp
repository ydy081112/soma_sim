#include "soma/hw/buffer.hpp"

#include <algorithm>

namespace soma {

ResourceReservation BufferResource::reserve(SimTime hw_arrival_time,
                                            SimTime hw_service_latency) {
    ResourceReservation result;
    result.hw_start_time = std::max(hw_arrival_time, hw_free_time_);
    result.hw_wait_latency = result.hw_start_time - hw_arrival_time;
    result.hw_finish_time = result.hw_start_time + hw_service_latency;
    hw_free_time_ = result.hw_finish_time;
    return result;
}

}  // namespace soma

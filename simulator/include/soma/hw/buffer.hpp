#pragma once

#include "soma/common/types.hpp"

namespace soma {

struct ResourceReservation {
    SimTime hw_start_time = 0;
    SimTime hw_finish_time = 0;
    SimTime hw_wait_latency = 0;
};

class BufferResource {
public:
    ResourceReservation reserve(SimTime hw_arrival_time, SimTime hw_service_latency);
    SimTime hw_free_time() const { return hw_free_time_; }

private:
    SimTime hw_free_time_ = 0;
};

}  // namespace soma

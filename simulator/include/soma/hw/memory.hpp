#pragma once

#include "soma/hw/buffer.hpp"

namespace soma {

class MemoryResource {
public:
    ResourceReservation read(SimTime hw_arrival_time, SimTime hw_latency);
    ResourceReservation write(SimTime hw_arrival_time, SimTime hw_latency);

private:
    BufferResource port_;
};

}  // namespace soma

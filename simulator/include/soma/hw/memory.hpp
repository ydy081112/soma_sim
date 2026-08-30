#pragma once

#include "soma/hw/hardware_resource.hpp"

namespace soma {

class MemoryResource {
public:
    ResourceReservation read(SimTime hw_arrival_time, SimTime hw_latency);
    ResourceReservation write(SimTime hw_arrival_time, SimTime hw_latency);

private:
    HardwareResource port_;
};

}  // namespace soma

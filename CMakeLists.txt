cmake_minimum_required(VERSION 3.16)
project(soma_sim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SOMA_BUILD_TESTS "Build focused unit tests" ON)

find_package(ZLIB REQUIRED)

add_library(soma_sim_lib
    simulator/src/config/simple_yaml.cpp
    simulator/src/config/hardware_config.cpp
    simulator/src/config/mapping_config.cpp
    simulator/src/runtime/npz.cpp
    simulator/src/runtime/weight_store.cpp
    simulator/src/runtime/spatial_template.cpp
    simulator/src/sim/spike_queue.cpp
    simulator/src/sim/stats.cpp
    simulator/src/hw/noc/route.cpp
    simulator/src/hw/noc/router.cpp
    simulator/src/hw/buffer.cpp
    simulator/src/hw/memory.cpp
    simulator/src/hw/synapse.cpp
    simulator/src/hw/soma.cpp
    simulator/src/hw/core.cpp
    simulator/src/input_encoder.cpp
    simulator/src/sim/simulator.cpp
)
target_include_directories(soma_sim_lib PUBLIC simulator/include)
target_link_libraries(soma_sim_lib PUBLIC ZLIB::ZLIB)
target_compile_options(soma_sim_lib PRIVATE -Wall -Wextra -Wpedantic)

add_executable(soma-sim simulator/src/main.cpp)
target_link_libraries(soma-sim PRIVATE soma_sim_lib)

if(SOMA_BUILD_TESTS)
    enable_testing()
    add_executable(soma-tests tests/test_simulator.cpp)
    target_link_libraries(soma-tests PRIVATE soma_sim_lib)
    add_test(NAME soma-focused-tests COMMAND soma-tests ${CMAKE_CURRENT_SOURCE_DIR})
endif()


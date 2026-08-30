#include "soma/sim/simulator.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --hardware PATH   hardware.yaml\n"
              << "  --mapping PATH    mapping.yaml\n"
              << "  --weights PATH    weights.npz\n"
              << "  --input PATH      input_spike.csv\n"
              << "  --output DIR      statistics output directory\n"
              << "  --max-events N    stop after N queue events (0 = unlimited)\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        soma::SimulatorOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage(argv[0]);
                return 0;
            }
            if (i + 1 >= argc) throw std::runtime_error("选项缺少值: " + arg);
            const std::string value = argv[++i];
            if (arg == "--hardware") options.hardware_path = value;
            else if (arg == "--mapping") options.mapping_path = value;
            else if (arg == "--weights") options.weights_path = value;
            else if (arg == "--input") options.input_spikes_path = value;
            else if (arg == "--output") options.output_dir = value;
            else if (arg == "--max-events") options.max_events = std::stoull(value);
            else throw std::runtime_error("未知选项: " + arg);
        }

        soma::Simulator simulator(std::move(options));
        const auto result = simulator.run();
        const auto prediction = result.output_scores.empty()
                                    ? -1
                                    : static_cast<int>(std::distance(
                                          result.output_scores.begin(),
                                          std::max_element(result.output_scores.begin(), result.output_scores.end())));
        std::cout << "SOMA-Sim " << (result.completed ? "completed" : "stopped at max-events")
                  << ": processed_spikes=" << result.processed_spikes
                  << ", hardware_latency_ps=" << result.hw_latency_ps
                  << ", host_latency_s=" << result.host_latency_s
                  << ", prediction=" << prediction << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "soma-sim error: " << error.what() << '\n';
        return 1;
    }
}

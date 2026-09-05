module {
  noc.network @noc0 {topology = "mesh", dimensions = [128, 256], hop_latency = 4.1ns, routing = "xy"}
  snn_arch.core_type @standard_core {neuron_capacity = 1024}
}

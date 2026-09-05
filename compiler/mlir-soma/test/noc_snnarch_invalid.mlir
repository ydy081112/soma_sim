module {
  noc.network @noc0 {
    topology = "mesh",
    dimensions = [0, 8],
    hop_latency = 2ns,
    routing = "xy"
  }
  snn_arch.core_type @invalid_core {
    neuron_capacity = 0
  }
}

module {
  noc.network @noc0 {
    topology = "mesh",
    dimensions = [8, 8],
    hop_latency = 2ns,
    routing = "xy"
  }
  snn_arch.core_type @standard_core {
    neuron_capacity = 1024,
    neuron_model = "st_bif"
  }
}

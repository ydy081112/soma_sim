module {
  noc.network @noc0 {topology = "mesh", dimensions = [1, 1], hop_latency = 1ns, routing = "xy"}
  snn_arch.core_type @standard_core {neuron_capacity = 4}
  snn_op.param @weight {source = "nir://test#/weight", kind = "weight", shape = array<i64: 6, 6>, dtype = "i8"}
  func.func @overflow(%arg0: tensor<1x6x!snn.spike<ternary>>) -> tensor<1x6x!snn.spike<ternary>> {
    %0 = snn_op.q_stbif %arg0 {in_features = 6 : i64, out_features = 6 : i64, weight = @weight, time_dim = 0 : i64, threshold = 3 : i16, tr_min = -7 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x6x!snn.spike<ternary>>) -> tensor<1x6x!snn.spike<ternary>>
    return %0 : tensor<1x6x!snn.spike<ternary>>
  }
}

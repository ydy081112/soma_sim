module {
  noc.network @noc0 {topology = "mesh", dimensions = [2, 2], hop_latency = 1ns, routing = "xy"}
  snn_arch.core_type @standard_core {neuron_capacity = 32}
  snn_op.param @q_weight {source = "nir://test#/q", kind = "weight", shape = array<i64: 2, 2>, dtype = "i8"}
  snn_op.param @k_weight {source = "nir://test#/k", kind = "weight", shape = array<i64: 2, 2>, dtype = "i8"}
  snn_op.param @v_weight {source = "nir://test#/v", kind = "weight", shape = array<i64: 2, 2>, dtype = "i8"}
  func.func @attention(%arg0: tensor<1x2x!snn.spike<ternary>>) -> tensor<1x2x!snn.spike<ternary>> {
    %0, %1 = snn_op.q_stbif %arg0 {in_features = 2 : i64, out_features = 2 : i64, weight = @q_weight, time_dim = 0 : i64, threshold = 3 : i16, tr_min = -7 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x2x!snn.spike<ternary>>) -> (tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>)
    %2, %3 = snn_op.k_stbif %arg0 {in_features = 2 : i64, out_features = 2 : i64, weight = @k_weight, time_dim = 0 : i64, threshold = 3 : i16, tr_min = -7 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x2x!snn.spike<ternary>>) -> (tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>)
    %4, %5 = snn_op.v_stbif %arg0 {in_features = 2 : i64, out_features = 2 : i64, weight = @v_weight, time_dim = 0 : i64, threshold = 3 : i16, tr_min = -7 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x2x!snn.spike<ternary>>) -> (tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>)
    %6, %7 = snn_op.qk_stbif %0, %1, %2, %3 {num_heads = 1 : i64, head_dim = 2 : i64, scale = 1.0 : f64, time_dim = 0 : i64, threshold = 3 : i16, tr_min = 0 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>, tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>) -> (tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>)
    %8 = snn_op.qkv_stbif %6, %7, %4, %5 {num_heads = 1 : i64, head_dim = 2 : i64, time_dim = 0 : i64, threshold = 3 : i16, tr_min = -7 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>, tensor<1x2x!snn.spike<ternary>>, tensor<1x2xi8>) -> tensor<1x2x!snn.spike<ternary>>
    %9 = snn_op.residual_stbif %arg0, %8 {w_main = 2 : i64, w_skip = 1 : i64, time_dim = 0 : i64, threshold = 3 : i16, tr_min = -7 : i8, tr_max = 7 : i8, voltage_dtype = "i16"} : (tensor<1x2x!snn.spike<ternary>>, tensor<1x2x!snn.spike<ternary>>) -> tensor<1x2x!snn.spike<ternary>>
    return %9 : tensor<1x2x!snn.spike<ternary>>
  }
}

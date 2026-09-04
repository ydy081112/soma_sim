// This exercises type-based fusion and the explicit Q/K/QKV tracer edges.
module {
  snn_op.param @w {dtype = "i8", kind = "weight", shape = array<i64: 4, 4>, source = "nir://unit#/w"}
  snn_op.param @b {dtype = "i32", kind = "bias", shape = array<i64: 4>, source = "nir://unit#/b"}

  func.func @fusion(%input: tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi8> {
    %q = snn_op.x_wq %input {bias = @b, in_features = 4 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @w} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi8>
    %q_st = snn_op.st_bif %q {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    %k = snn_op.x_wk %input {bias = @b, in_features = 4 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @w} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi8>
    %k_st = snn_op.st_bif %k {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    %v = snn_op.x_wv %input {bias = @b, in_features = 4 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @w} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi8>
    %v_st = snn_op.st_bif %v {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    %qk = snn_op.qk %q_st, %k_st {head_dim = 4 : i64, num_heads = 1 : i64, scale = 1.0 : f64, time_dim = 0 : i64} : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>
    %qk_st = snn_op.st_bif %qk {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    %qkv = snn_op.qkv %qk_st, %v_st {head_dim = 4 : i64, num_heads = 1 : i64, time_dim = 0 : i64} : (tensor<2x4xi8>, tensor<2x4xi8>) -> tensor<2x4xi8>
    %qkv_st = snn_op.st_bif %qkv {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    %residual = snn_op.residual %input, %qkv_st {time_dim = 0 : i64, w_main = 2 : i64, w_skip = 4 : i64} : (tensor<2x4x!snn.spike<ternary>>, tensor<2x4xi8>) -> tensor<2x4xi8>
    %out = snn_op.st_bif %residual {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    // Both results of this fused pair are dead and must be removed by
    // --dead-neuron-out-eliminate.
    %dead = snn_op.linear %input {bias = @b, in_features = 4 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @w} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi8>
    %dead_st = snn_op.st_bif %dead {threshold = 1 : i16, time_dim = 0 : i64, tr_max = 7 : i8, tr_min = -7 : i8, voltage_dtype = "i16"} : tensor<2x4xi8> -> tensor<2x4xi8>
    return %out : tensor<2x4xi8>
  }
}

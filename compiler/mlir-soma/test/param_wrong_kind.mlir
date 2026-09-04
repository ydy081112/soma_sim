module {
  snn_op.param @not_weight {dtype = "i32", kind = "bias", shape = array<i64: 4>, source = "nir://unit#/bias"}
  func.func @wrong_kind(%input: tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32> {
    %0 = snn_op.linear %input {in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @not_weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }
}

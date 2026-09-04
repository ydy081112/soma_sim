module {
  func.func @missing(%input: tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32> {
    %0 = snn_op.linear %input {in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @does_not_exist} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }
}

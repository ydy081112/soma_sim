module {
  func.func @canonicalize(%x: tensor<2x!snn.spike<ternary>>) -> tensor<2x!snn.spike<ternary>> {
    %0 = snn_op.rescale %x {scale = 2.0 : f64, time_dim = 0 : i64} : (tensor<2x!snn.spike<ternary>>) -> tensor<2x!snn.spike<ternary>>
    %1 = snn_op.rescale %0 {scale = 0.5 : f64, time_dim = 0 : i64} : (tensor<2x!snn.spike<ternary>>) -> tensor<2x!snn.spike<ternary>>
    return %1 : tensor<2x!snn.spike<ternary>>
  }
}

module {
  func.func @missing_weight(%input: tensor<2x4x!snn.spike<ternary>>)
      -> tensor<2x4x!snn.spike<ternary>> {
    %0 = snn_exec.generic
        ins(%input : tensor<2x4x!snn.spike<ternary>>)
        -> (tensor<2x4x!snn.spike<ternary>>)
        attributes {time_dim = 0 : i64} {
      %1 = snn_exec.state : !snn.state<tensor<4x!snn.voltage<i16>>>
      %2 = snn_exec.sw %input {weight = @missing} :
        tensor<2x4x!snn.spike<ternary>> -> tensor<4x!snn.voltage<i16>>
      %3 = snn_exec.integrate %1, %2 :
        (!snn.state<tensor<4x!snn.voltage<i16>>>,
         tensor<4x!snn.voltage<i16>>) -> tensor<4x!snn.voltage<i16>>
      %4 = snn_exec.fire %3 {
        threshold = 1 : i16,
        tr_max = 7 : i8,
        tr_min = -7 : i8
      } : (tensor<4x!snn.voltage<i16>>) -> tensor<4x!snn.spike<ternary>>
      snn_exec.yield %4 : tensor<4x!snn.spike<ternary>>
    }
    return %0 : tensor<2x4x!snn.spike<ternary>>
  }
}

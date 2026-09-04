module {
  snn_op.param @scale {
    source = "nir://unit#/scale",
    kind = "weight",
    shape = array<i64: 4>,
    dtype = "i8"
  }

  func.func @execute(%input: tensor<2x4x!snn.spike<ternary>>)
      -> (tensor<2x4x!snn.spike<ternary>>, tensor<2x4xi8>) {
    %0, %1 = snn_exec.generic
        ins(%input : tensor<2x4x!snn.spike<ternary>>)
        -> (tensor<2x4x!snn.spike<ternary>>, tensor<2x4xi8>)
        attributes {time_dim = 0 : i64} {
      %2 = snn_exec.state : !snn.state<tensor<4x!snn.voltage<i16>>>
      %3 = snn_exec.mul %input {
        axis = -1 : i64,
        weight = @scale
      } : tensor<2x4x!snn.spike<ternary>> -> tensor<4x!snn.voltage<i16>>
      %4 = snn_exec.reduce %input {
        kernel = "1",
        kind = "avg",
        stride = "1"
      } : tensor<2x4x!snn.spike<ternary>> -> tensor<4x!snn.voltage<i16>>
      %5 = snn_exec.integrate %2, %3, %4 :
        (!snn.state<tensor<4x!snn.voltage<i16>>>,
         tensor<4x!snn.voltage<i16>>,
         tensor<4x!snn.voltage<i16>>) -> tensor<4x!snn.voltage<i16>>
      %6, %7 = snn_exec.fire %5 {
        threshold = 398 : i16,
        tr_max = 7 : i8,
        tr_min = -7 : i8
      } : (tensor<4x!snn.voltage<i16>>) ->
          (tensor<4x!snn.spike<ternary>>, tensor<4xi8>)
      snn_exec.yield %6, %7 : tensor<4x!snn.spike<ternary>>, tensor<4xi8>
    }
    return %0, %1 : tensor<2x4x!snn.spike<ternary>>, tensor<2x4xi8>
  }
}

module {
  snn_op.param @weight {dtype = "i8", kind = "weight", shape = array<i64: 4>, source = "nir://unit#/weight"}
  snn_op.param @bias {dtype = "i32", kind = "bias", shape = array<i64: 4>, source = "nir://unit#/bias"}

  func.func @bad_affine(%input: tensor<2x4x!snn.spike<ternary>>) -> tensor<2x5xi32> {
    %0 = snn_op.affine %input {axis = -1 : i64, bias = @bias, time_dim = 0 : i64, weight = @weight} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x5xi32>
    return %0 : tensor<2x5xi32>
  }
}

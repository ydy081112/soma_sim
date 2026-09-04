module {
  snn_op.param @weight {dtype = "i8", kind = "weight", shape = array<i64: 4, 3>, source = "nir://unit#/weight"}
  snn_op.param @bias {dtype = "i32", kind = "bias", shape = array<i64: 4>, source = "nir://unit#/bias"}
  snn_op.param @conv_weight {dtype = "i8", kind = "weight", shape = array<i64: 4, 3, 3, 3>, source = "nir://unit#/conv_weight"}
  snn_op.param @affine_weight {dtype = "i8", kind = "weight", shape = array<i64: 4>, source = "nir://unit#/affine_weight"}

  func.func @with_bias(%input: tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32> {
    %0 = snn_op.linear %input {bias = @bias, in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }

  func.func @without_bias(%input: tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32> {
    %0 = snn_op.linear %input {in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }

  func.func @conv(%input: tensor<1x3x4x4x!snn.spike<binary>>) -> tensor<1x4x2x2xi32> {
    %0 = snn_op.conv2d %input {groups = 1 : i64, kernel = "3,3", padding = "0,0", stride = "1,1", time_dim = 0 : i64, weight = @conv_weight} : (tensor<1x3x4x4x!snn.spike<binary>>) -> tensor<1x4x2x2xi32>
    return %0 : tensor<1x4x2x2xi32>
  }

  func.func @affine(%input: tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi32> {
    %0 = snn_op.affine %input {axis = -1 : i64, bias = @bias, time_dim = 0 : i64, weight = @affine_weight} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi32>
    return %0 : tensor<2x4xi32>
  }

  func.func @specialized(%input: tensor<2x3x!snn.spike<ternary>>, %norm_input: tensor<2x4x!snn.spike<ternary>>, %attention: tensor<1x2x!snn.spike<ternary>>, %value: tensor<1x2x!snn.spike<ternary>>) -> tensor<2x4xi32> {
    %q = snn_op.x_wq %input {bias = @bias, in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    %k = snn_op.x_wk %input {bias = @bias, in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    %v = snn_op.x_wv %input {bias = @bias, in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    %o = snn_op.z_wo %input {bias = @bias, in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    %fc = snn_op.fc %input {bias = @bias, in_features = 3 : i64, out_features = 4 : i64, time_dim = 0 : i64, weight = @weight} : (tensor<2x3x!snn.spike<ternary>>) -> tensor<2x4xi32>
    %norm = snn_op.norm %norm_input {axis = -1 : i64, bias = @bias, time_dim = 0 : i64, weight = @affine_weight} : (tensor<2x4x!snn.spike<ternary>>) -> tensor<2x4xi32>
    %qkv = snn_op.qkv %attention, %value {head_dim = 2 : i64, num_heads = 1 : i64, time_dim = 0 : i64} : (tensor<1x2x!snn.spike<ternary>>, tensor<1x2x!snn.spike<ternary>>) -> tensor<1x2xi32>
    return %q : tensor<2x4xi32>
  }
}

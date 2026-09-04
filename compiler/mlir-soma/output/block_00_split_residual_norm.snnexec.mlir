module attributes {snn_op.frontend = "nir-1.0.8", snn_op.weight_source = "nir"} {
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/norm1_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/norm1_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_q_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128, 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/q_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_q_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/q_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_k_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128, 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/k_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_k_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/k_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_v_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128, 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/v_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_v_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/v_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_proj_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128, 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/proj_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_proj_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/proj_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm2_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/norm2_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm2_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/norm2_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc1_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 512, 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/fc1_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc1_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 512>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/fc1_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc2_IF_weight {dtype = "int8", kind = "weight", shape = array<i64: 128, 512>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/fc2_IF/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc2_IF_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/fc2_IF/metadata/bias"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_next_weight {dtype = "int8", kind = "weight", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/norm1_IF_next/metadata/weight"}
  snn_op.param @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_next_bias {dtype = "int16", kind = "bias", shape = array<i64: 128>, source = "nir://block_00_until_silent_t30_cls_free_split_residual_norm.nir#/node/nodes/norm1_IF_next/metadata/bias"}
  func.func @block(%arg0: tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>, tensor<30x1x64x128xi8>) {
    %0 = snn_exec.generic ins(%arg0 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_bias} : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.mul %arg0 {axis = -1 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %21 = snn_exec.fire %20 {threshold = 398 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.spike<ternary>>
      snn_exec.yield %21 : tensor<1x64x128x!snn.spike<ternary>>
    }
    %1, %2 = snn_exec.generic ins(%0 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_q_IF_bias} : !snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>
      %19 = snn_exec.sw %0 {head_dim = 64 : i64, in_features = 128 : i64, num_heads = 2 : i64, out_features = 128 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_q_IF_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x2x64x64x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>, tensor<1x2x64x64x!snn.voltage<i16>>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %21, %22 = snn_exec.fire %20 {threshold = 699 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x2x64x64x!snn.voltage<i16>>) -> (tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>)
      snn_exec.yield %21, %22 : tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>
    }
    %3, %4 = snn_exec.generic ins(%0 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_k_IF_bias} : !snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>
      %19 = snn_exec.sw %0 {head_dim = 64 : i64, in_features = 128 : i64, num_heads = 2 : i64, out_features = 128 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_k_IF_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x2x64x64x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>, tensor<1x2x64x64x!snn.voltage<i16>>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %21, %22 = snn_exec.fire %20 {threshold = 531 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x2x64x64x!snn.voltage<i16>>) -> (tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>)
      snn_exec.yield %21, %22 : tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>
    }
    %5, %6 = snn_exec.generic ins(%0 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_v_IF_bias} : !snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>
      %19 = snn_exec.sw %0 {head_dim = 64 : i64, in_features = 128 : i64, num_heads = 2 : i64, out_features = 128 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_v_IF_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x2x64x64x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>, tensor<1x2x64x64x!snn.voltage<i16>>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %21, %22 = snn_exec.fire %20 {threshold = 669 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x2x64x64x!snn.voltage<i16>>) -> (tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>)
      snn_exec.yield %21, %22 : tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>
    }
    %7, %8 = snn_exec.generic ins(%1, %2, %3, %4 : tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>, tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) -> (tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state : !snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>
      %19 = snn_exec.ss %1, %2, %3, %4 {head_dim = 64 : i64, num_heads = 2 : i64, scale = 1.000000e+00 : f64} : (tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>, tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>, tensor<1x2x64x64x!snn.voltage<i16>>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %21, %22 = snn_exec.fire %20 {threshold = 656 : i16, tr_max = 7 : i8, tr_min = 0 : i8} : (tensor<1x2x64x64x!snn.voltage<i16>>) -> (tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>)
      snn_exec.yield %21, %22 : tensor<1x2x64x64x!snn.spike<ternary>>, tensor<1x2x64x64xi8>
    }
    %9 = snn_exec.generic ins(%7, %8, %5, %6 : tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>, tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) -> (tensor<30x1x2x64x64x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state : !snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>
      %19 = snn_exec.ss %7, %8, %5, %6 {head_dim = 64 : i64, num_heads = 2 : i64} : (tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>, tensor<30x1x2x64x64x!snn.spike<ternary>>, tensor<30x1x2x64x64xi8>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x2x64x64x!snn.voltage<i16>>>, tensor<1x2x64x64x!snn.voltage<i16>>) -> tensor<1x2x64x64x!snn.voltage<i16>>
      %21 = snn_exec.fire %20 {threshold = 646 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x2x64x64x!snn.voltage<i16>>) -> tensor<1x2x64x64x!snn.spike<ternary>>
      snn_exec.yield %21 : tensor<1x2x64x64x!snn.spike<ternary>>
    }
    %10 = snn_exec.generic ins(%9 : tensor<30x1x2x64x64x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_proj_IF_bias} : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.sw %9 {in_features = 128 : i64, out_features = 128 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_proj_IF_weight} : tensor<30x1x2x64x64x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %21 = snn_exec.fire %20 {threshold = 561 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.spike<ternary>>
      snn_exec.yield %21 : tensor<1x64x128x!snn.spike<ternary>>
    }
    %11 = snn_exec.generic ins(%arg0, %10 : tensor<30x1x64x128x!snn.spike<ternary>>, tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.sw %arg0 {weight = 512 : i64} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.sw %10 {weight = 128 : i64} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %21 = snn_exec.integrate %18, %19, %20 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %22 = snn_exec.fire %21 {threshold = 589 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.spike<ternary>>
      snn_exec.yield %22 : tensor<1x64x128x!snn.spike<ternary>>
    }
    %12 = snn_exec.generic ins(%11 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm2_IF_bias} : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.mul %11 {axis = -1 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm2_IF_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %21 = snn_exec.fire %20 {threshold = 515 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.spike<ternary>>
      snn_exec.yield %21 : tensor<1x64x128x!snn.spike<ternary>>
    }
    %13 = snn_exec.generic ins(%12 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x512x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc1_IF_bias} : !snn.state<tensor<1x64x512x!snn.voltage<i16>>>
      %19 = snn_exec.sw %12 {in_features = 128 : i64, out_features = 512 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc1_IF_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x512x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x64x512x!snn.voltage<i16>>>, tensor<1x64x512x!snn.voltage<i16>>) -> tensor<1x64x512x!snn.voltage<i16>>
      %21 = snn_exec.fire %20 {threshold = 703 : i16, tr_max = 7 : i8, tr_min = 0 : i8} : (tensor<1x64x512x!snn.voltage<i16>>) -> tensor<1x64x512x!snn.spike<ternary>>
      snn_exec.yield %21 : tensor<1x64x512x!snn.spike<ternary>>
    }
    %14 = snn_exec.generic ins(%13 : tensor<30x1x64x512x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc2_IF_bias} : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.sw %13 {in_features = 512 : i64, out_features = 128 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_fc2_IF_weight} : tensor<30x1x64x512x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %21 = snn_exec.fire %20 {threshold = 843 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.spike<ternary>>
      snn_exec.yield %21 : tensor<1x64x128x!snn.spike<ternary>>
    }
    %15 = snn_exec.generic ins(%11, %14 : tensor<30x1x64x128x!snn.spike<ternary>>, tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.sw %11 {weight = 512 : i64} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.sw %14 {weight = 256 : i64} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %21 = snn_exec.integrate %18, %19, %20 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %22 = snn_exec.fire %21 {threshold = 718 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.spike<ternary>>
      snn_exec.yield %22 : tensor<1x64x128x!snn.spike<ternary>>
    }
    %16, %17 = snn_exec.generic ins(%15 : tensor<30x1x64x128x!snn.spike<ternary>>) -> (tensor<30x1x64x128x!snn.spike<ternary>>, tensor<30x1x64x128xi8>) attributes {time_dim = 0 : i64} {
      %18 = snn_exec.state {init = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_next_bias} : !snn.state<tensor<1x64x128x!snn.voltage<i16>>>
      %19 = snn_exec.mul %15 {axis = -1 : i64, weight = @b00_block_00_until_silent_t30_cls_free_split_residual_norm_norm1_IF_next_weight} : tensor<30x1x64x128x!snn.spike<ternary>> -> tensor<1x64x128x!snn.voltage<i16>>
      %20 = snn_exec.integrate %18, %19 : (!snn.state<tensor<1x64x128x!snn.voltage<i16>>>, tensor<1x64x128x!snn.voltage<i16>>) -> tensor<1x64x128x!snn.voltage<i16>>
      %21, %22 = snn_exec.fire %20 {threshold = 655 : i16, tr_max = 7 : i8, tr_min = -7 : i8} : (tensor<1x64x128x!snn.voltage<i16>>) -> (tensor<1x64x128x!snn.spike<ternary>>, tensor<1x64x128xi8>)
      snn_exec.yield %21, %22 : tensor<1x64x128x!snn.spike<ternary>>, tensor<1x64x128xi8>
    }
    return %16, %17 : tensor<30x1x64x128x!snn.spike<ternary>>, tensor<30x1x64x128xi8>
  }
}


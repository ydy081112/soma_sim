snn-exec
主要目标：这个dialect作为承接snn-op的下一层方言，没有任何snn模型层面算子的原语了（比如qk这种），需要全部变成面向spike的计算op

1. snn_exec.generic：
在snn-exec层，首先我的每一个snn-op的op会变成一个snn_exec.generic构成的region；
snn_exec.generic的ins和outs都是完全复刻上层snn-op；
在snn_exec.generic的内部最开始需要先创建一个snn_exec.state，形状是输出神经元层的形状，他代表每个神经元的membrane；
大概长这样子：  %mb = snn_exec.state {   init = @bias   } : !snn.state<tensor<1x2x64x64xi16>> 这样子，snn_exec.state 虽然写在 generic region 内，但语义上是 persistent state declaration，不是 generic 每次被 spike 触发时重新创建一次，所以你还需要创建一个state type

2. linalg.generic的block内部的op：
首先两个重要的op是snn_exec.sw和snn_exec.ss，就是spike乘weight，和，spike乘spike
然后就是snn_exec.integrate以及snn_exec.fire

我列几个从snn-op的op，编译到snn-exec的op：

2.1 q_stbif编译成：
snn_exec.sw--他的输入就是一个spike类型，然后输出是一个电压类型，这个需要你去创建type：比如这样子：tensor<1x64x64x!snn.voltage<i16>>
snn_exec.integrate--他的输入是刚才创建的snn_exec.state的输出（也就是membrane）和snn_exec.sw的spike类型的输出，他的输出也是一个电压类型
snn_exec.fire--这个是用来表示发放spike的算子，他的输入是是snn_exec.integrate输出的电压类型，原来snn-op的threshold的参数放在这个op的参数里面，然后输出是spike和tracer
2.2 residual_stbif编译成：
snn_exec.sw--这个op在residual里面会有两个，然后weight直接复用snn-op的w_main和w_skip，输出也是电压类型
snn_exec.integrate--这个是用来累加的，他的input有三个“mb和snn_exec.sw的spike类型的output”
snn_exec.fire--同2.1
2.3 qk_stbif/qkv_stbif编译成：
snn_exec.ss--他的输入是两个spike类型以及两个tracer，然后输出也是一个电压类型
snn_exec.integrate--同2.1 2.2
snn_exec.fire--同2.1 2.2
2.4 norm_stbif编译成：
snn_exec.mul--也就是输入还是spike类型，然后乘他对应的权重（不是矩阵乘这里，相当于逐元素乘）
snn_exec.integrate--同2.1 2.2
snn_exec.fire--同2.1 2.2


下面是例子：
%q_spike, %q_tracer = snn_exec.generic
    ins(%input : tensor<30x1x65x128x!snn.spike<ternary>>)
    outs(%output:  tensor<30x1x4x65x32x!snn.spike<ternary>>, tensor<30x1x4x65x32xi16>)
    {
     %mb = snn_exec.state { init = @q_bias } : !snn.state<tensor<1x4x65x32x!snn.voltage<i16>>>

      // 当前 spike 对目标神经元产生的电压贡献：
      //
      // delta_V = spike × weight
      //
      // sw 内部知道 @q_weight 以及 linear projection 的索引关系。
      %delta = snn_exec.sw %input {
          in_features = 128 : i64,
          out_features = 128 : i64,
          num_heads = 4 : i64,
          head_dim = 32 : i64
       weight = @weight
      } :
        tensor<30x1x65x128x!snn.spike<ternary>>
        ->
        tensor<1x4x65x32x!snn.voltage<i16>>

      // V = membrane + delta_V
      %v = snn_exec.integrate %mb, %delta :
        (
          !snn.state<tensor<1x4x65x32x!snn.voltage<i16>>>,
          tensor<1x4x65x32x!snn.voltage<i16>>
        ) -> tensor<1x4x65x32x!snn.voltage<i16>>

      // ST-BIF firing。
      //
      // 根据 threshold 判断发放，
      // 同时产生 spike tracer。
      %spike, %tracer =
          snn_exec.fire %v {
              threshold = 7564 : i64,
              tr_min = -7 : i64,
              tr_max = 7 : i64
          } :
          tensor<1x4x65x32x!snn.voltage<i16>>
          ->
          (
            tensor<1x4x65x32x!snn.spike<ternary>>,
            tensor<1x4x65x32xi16>
          )
    snn_exec.yield %spike, %tracer
    }
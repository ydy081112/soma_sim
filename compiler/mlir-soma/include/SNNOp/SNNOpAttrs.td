// V1 属性由每个 op 的具名 Attr 约束定义：neuron_model、threshold、tau、time_dim。
// spike encoding 从输入 tensor type 推导；这样文本 IR 的属性归属明确，后续可无损升级为
// 更紧凑的 dialect attribute，而不改变 op 语义。

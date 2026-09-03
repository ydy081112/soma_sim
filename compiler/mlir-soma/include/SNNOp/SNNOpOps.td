include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SNNOp/SNNOpDialect.td"
include "SNNOp/SNNOpAttrs.td"

class SNNOp_Op<string mnemonic, list<Trait> traits = []> : Op<SNNOpDialect, mnemonic, traits>;

def Conv2DOp : SNNOp_Op<"conv2d", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kernel, StrAttr:$stride,
                   StrAttr:$padding, I64Attr:$groups, StrAttr:$weight, OptionalAttr<StrAttr>:$bias,
                   StrAttr:$spike_encoding, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def LinearOp : SNNOp_Op<"linear", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$in_features, I64Attr:$out_features,
                   StrAttr:$weight, OptionalAttr<StrAttr>:$bias, StrAttr:$spike_encoding, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def QKOp : SNNOp_Op<"qk", [Pure]> {
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key, I64Attr:$num_heads,
                   I64Attr:$head_dim, F64Attr:$scale, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$query `,` $key attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def AVOp : SNNOp_Op<"av", [Pure]> {
  let arguments = (ins AnyRankedTensor:$attention, AnyRankedTensor:$value, I64Attr:$num_heads,
                   I64Attr:$head_dim, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$attention `,` $value attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def ResidualOp : SNNOp_Op<"residual", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def PoolOp : SNNOp_Op<"pool", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kind, StrAttr:$kernel, StrAttr:$stride, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def RescaleOp : SNNOp_Op<"rescale", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, F64Attr:$scale, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def LIFOp : SNNOp_Op<"lif", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, F64Attr:$threshold, F64Attr:$tau,
                   StrAttr:$reset_mode, F64Attr:$reset_value, StrAttr:$neuron_model, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def STBIFOp : SNNOp_Op<"st_bif", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, F64Attr:$threshold, F64Attr:$tr_min,
                   F64Attr:$tr_max, StrAttr:$neuron_model, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}

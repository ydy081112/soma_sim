include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "SNNOp/SNNOpDialect.td"
include "SNNOp/SNNOpAttrs.td"

class SNNOp_Op<string mnemonic, list<Trait> traits = []> : Op<SNNOpDialect, mnemonic, traits>;

def ParamOp : SNNOp_Op<"param", [Symbol, HasParent<"::mlir::ModuleOp">]> {
  let summary = "不嵌入内容的不可变模型参数声明";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$source, StrAttr:$kind,
                   DenseI64ArrayAttr:$shape, StrAttr:$dtype);
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def Conv2DOp : SNNOp_Op<"conv2d", [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kernel, StrAttr:$stride,
                   StrAttr:$padding, I64Attr:$groups, FlatSymbolRefAttr:$weight,
                   OptionalAttr<FlatSymbolRefAttr>:$bias, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
class LinearLikeOp<string mnemonic> : SNNOp_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$in_features, I64Attr:$out_features,
                   FlatSymbolRefAttr:$weight, OptionalAttr<FlatSymbolRefAttr>:$bias, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def LinearOp : LinearLikeOp<"linear">;
def XWQOp : LinearLikeOp<"x_wq">;
def XWKOp : LinearLikeOp<"x_wk">;
def XWVOp : LinearLikeOp<"x_wv">;
def ZWOOp : LinearLikeOp<"z_wo">;
def FCOp : LinearLikeOp<"fc">;

class AffineLikeOp<string mnemonic> : SNNOp_Op<mnemonic, [Pure]> {
  let summary = "按指定轴执行逐通道 weight/bias 仿射变换";
  let arguments = (ins AnyRankedTensor:$input, FlatSymbolRefAttr:$weight,
                   FlatSymbolRefAttr:$bias, I64Attr:$axis, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}
def AffineOp : AffineLikeOp<"affine">;
def NormOp : AffineLikeOp<"norm">;
def QKOp : SNNOp_Op<"qk", [Pure]> {
  let arguments = (ins AnyRankedTensor:$query, AnyRankedTensor:$key, I64Attr:$num_heads,
                   I64Attr:$head_dim, F64Attr:$scale, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$query `,` $key attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
class AVLikeOp<string mnemonic> : SNNOp_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$attention, AnyRankedTensor:$value, I64Attr:$num_heads,
                   I64Attr:$head_dim, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$attention `,` $value attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def AVOp : AVLikeOp<"av">;
def QKVOp : AVLikeOp<"qkv">;
def ResidualOp : SNNOp_Op<"residual", [Pure]> {
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs, I64Attr:$w_main,
                   I64Attr:$w_skip, I64Attr:$time_dim);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
class FusedSTBIFOp<string mnemonic> : SNNOp_Op<mnemonic, [Pure, OpAsmOpInterface]> {
  let extraClassDeclaration = [{
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  }];
}
class LinearSTBIFLikeOp<string mnemonic> : FusedSTBIFOp<mnemonic> {
  let arguments = (ins AnyRankedTensor:$input, I64Attr:$in_features, I64Attr:$out_features,
                   FlatSymbolRefAttr:$weight, OptionalAttr<FlatSymbolRefAttr>:$bias,
                   I64Attr:$time_dim, AnyAttr:$threshold, AnyAttr:$tr_min,
                   AnyAttr:$tr_max, StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}
def LinearSTBIFOp : LinearSTBIFLikeOp<"linear_stbif">;
def QSTBIFOp : LinearSTBIFLikeOp<"q_stbif">;
def KSTBIFOp : LinearSTBIFLikeOp<"k_stbif">;
def VSTBIFOp : LinearSTBIFLikeOp<"v_stbif">;
def ZSTBIFOp : LinearSTBIFLikeOp<"z_stbif">;
def FCSTBIFOp : LinearSTBIFLikeOp<"fc_stbif">;

class AffineSTBIFLikeOp<string mnemonic> : FusedSTBIFOp<mnemonic> {
  let arguments = (ins AnyRankedTensor:$input, FlatSymbolRefAttr:$weight,
                   FlatSymbolRefAttr:$bias, I64Attr:$axis, I64Attr:$time_dim,
                   AnyAttr:$threshold, AnyAttr:$tr_min, AnyAttr:$tr_max,
                   StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}
def AffineSTBIFOp : AffineSTBIFLikeOp<"affine_stbif">;
def NormSTBIFOp : AffineSTBIFLikeOp<"norm_stbif">;

def Conv2DSTBIFOp : FusedSTBIFOp<"conv2d_stbif"> {
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kernel, StrAttr:$stride,
                   StrAttr:$padding, I64Attr:$groups, FlatSymbolRefAttr:$weight,
                   OptionalAttr<FlatSymbolRefAttr>:$bias, I64Attr:$time_dim,
                   AnyAttr:$threshold, AnyAttr:$tr_min, AnyAttr:$tr_max,
                   StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}
def QKSTBIFOp : FusedSTBIFOp<"qk_stbif"> {
  let arguments = (ins AnyRankedTensor:$query_spike, AnyRankedTensor:$query_tracer,
                   AnyRankedTensor:$key_spike, AnyRankedTensor:$key_tracer,
                   I64Attr:$num_heads, I64Attr:$head_dim, F64Attr:$scale,
                   I64Attr:$time_dim, AnyAttr:$threshold, AnyAttr:$tr_min,
                   AnyAttr:$tr_max, StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$query_spike `,` $query_tracer `,` $key_spike `,` $key_tracer attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def QKVSTBIFOp : FusedSTBIFOp<"qkv_stbif"> {
  let arguments = (ins AnyRankedTensor:$attention_spike, AnyRankedTensor:$attention_tracer,
                   AnyRankedTensor:$value_spike, AnyRankedTensor:$value_tracer,
                   I64Attr:$num_heads, I64Attr:$head_dim, I64Attr:$time_dim,
                   AnyAttr:$threshold, AnyAttr:$tr_min, AnyAttr:$tr_max,
                   StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$attention_spike `,` $attention_tracer `,` $value_spike `,` $value_tracer attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def ResidualSTBIFOp : FusedSTBIFOp<"residual_stbif"> {
  let arguments = (ins AnyRankedTensor:$main_spike, AnyRankedTensor:$skip_spike,
                   I64Attr:$w_main, I64Attr:$w_skip, I64Attr:$time_dim,
                   AnyAttr:$threshold, AnyAttr:$tr_min, AnyAttr:$tr_max,
                   StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$main_spike `,` $skip_spike attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}
def RescaleSTBIFOp : FusedSTBIFOp<"rescale_stbif"> {
  let arguments = (ins AnyRankedTensor:$input, F64Attr:$scale, I64Attr:$time_dim,
                   AnyAttr:$threshold, AnyAttr:$tr_min, AnyAttr:$tr_max,
                   StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
}
def PoolSTBIFOp : FusedSTBIFOp<"pool_stbif"> {
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kind, StrAttr:$kernel,
                   StrAttr:$stride, I64Attr:$time_dim, AnyAttr:$threshold,
                   AnyAttr:$tr_min, AnyAttr:$tr_max, StrAttr:$voltage_dtype);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
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
  let arguments = (ins AnyRankedTensor:$input, AnyAttr:$threshold, AnyAttr:$tr_min,
                   AnyAttr:$tr_max, I64Attr:$time_dim, StrAttr:$voltage_dtype);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

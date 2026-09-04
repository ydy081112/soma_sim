include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SNNExec/SNNExecDialect.td"

class SNNExec_Op<string mnemonic, list<Trait> traits = []>
    : Op<SNNExecDialect, mnemonic, traits>;

def GenericOp : SNNExec_Op<"generic", [SingleBlock, OpAsmOpInterface]> {
  let summary = "一个 fused neuron population 的逐 timestep 执行 region";
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs, I64Attr:$time_dim);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "`ins` `(` $inputs `:` type($inputs) `)` `->` `(` type($outputs) `)` attr-dict-with-keyword $body";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  }];
}

def StateOp : SNNExec_Op<"state"> {
  let summary = "persistent membrane state declaration";
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$init);
  let results = (outs AnyType:$state);
  let assemblyFormat = "attr-dict `:` type($state)";
  let hasVerifier = 1;
}

def SWOp : SNNExec_Op<"sw", [Pure]> {
  let summary = "spike times symbolic or scalar weight";
  let arguments = (ins AnyRankedTensor:$input, AnyAttr:$weight);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def SSOp : SNNExec_Op<"ss", [Pure]> {
  let summary = "two spike/tracer populations contribute a product voltage";
  let arguments = (ins AnyRankedTensor:$lhs_spike, AnyRankedTensor:$lhs_tracer,
                   AnyRankedTensor:$rhs_spike, AnyRankedTensor:$rhs_tracer);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$lhs_spike `,` $lhs_tracer `,` $rhs_spike `,` $rhs_tracer attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def MulOp : SNNExec_Op<"mul", [Pure]> {
  let summary = "elementwise spike multiplication";
  let arguments = (ins AnyRankedTensor:$input, AnyAttr:$weight);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def ReduceOp : SNNExec_Op<"reduce", [Pure]> {
  let summary = "windowed spike reduction";
  let arguments = (ins AnyRankedTensor:$input, StrAttr:$kind, StrAttr:$kernel,
                   StrAttr:$stride);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

def IntegrateOp : SNNExec_Op<"integrate", [Pure]> {
  let summary = "add voltage contributions to persistent membrane state";
  let arguments = (ins AnyType:$state, Variadic<AnyRankedTensor>:$deltas);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$state `,` $deltas attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
}

def FireOp : SNNExec_Op<"fire", [OpAsmOpInterface]> {
  let summary = "threshold and emit the live spike/tracer results";
  let arguments = (ins AnyRankedTensor:$input, AnyAttr:$threshold,
                   AnyAttr:$tr_min, AnyAttr:$tr_max);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, results)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  }];
}

def YieldOp : SNNExec_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyRankedTensor>:$values);
  let assemblyFormat = "$values attr-dict `:` type($values)";
  let hasVerifier = 1;
}

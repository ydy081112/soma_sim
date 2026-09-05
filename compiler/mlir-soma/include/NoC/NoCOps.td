include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "NoC/NoCDialect.td"

class NoC_Op<string mnemonic, list<Trait> traits = []>
    : Op<NoCDialect, mnemonic, traits>;

def NetworkOp : NoC_Op<"network", [Symbol, HasParent<"::mlir::ModuleOp">]> {
  let summary = "module-level core network declaration";
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$topology,
                   DenseI64ArrayAttr:$dimensions, AnyAttr:$hop_latency,
                   StrAttr:$routing);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

class RouterOp<string mnemonic> : NoC_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyRankedTensor:$input, FlatSymbolRefAttr:$network,
                   DenseI64ArrayAttr:$coord);
  let results = (outs AnyRankedTensor:$output);
  let assemblyFormat = "$input attr-dict `:` functional-type($input, $output)";
  let hasVerifier = 1;
}

def SendRouterOp : RouterOp<"send_router">;
def RecvRouterOp : RouterOp<"recv_router">;

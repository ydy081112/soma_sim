include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "SNNArch/SNNArchDialect.td"

class SNNArch_Op<string mnemonic, list<Trait> traits = []>
    : Op<SNNArchDialect, mnemonic, traits>;

def CoreTypeOp : SNNArch_Op<"core_type", [Symbol, HasParent<"::mlir::ModuleOp">]> {
  let summary = "module-level core type declaration";
  let arguments = (ins SymbolNameAttr:$sym_name, I64Attr:$neuron_capacity,
                   OptionalAttr<StrAttr>:$neuron_model);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

class ModelCoreOp<string mnemonic> : SNNArch_Op<mnemonic, [Pure, OpAsmOpInterface]> {
  let summary = "partitioned model-aware core operation";
  // These parallel arrays describe every variadic tensor operand.  The
  // source_operand entry identifies the logical operand of the fused model op.
  let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                   FlatSymbolRefAttr:$core_type, I64Attr:$core_id,
                   DenseI64ArrayAttr:$coord, I64Attr:$partition_id,
                   I64Attr:$partition_offset, I64Attr:$partition_size,
                   DenseI64ArrayAttr:$source_operand,
                   DenseI64ArrayAttr:$source_partition,
                   DenseI64ArrayAttr:$source_offset,
                   DenseI64ArrayAttr:$source_size);
  let results = (outs Variadic<AnyRankedTensor>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  }];
}

def Conv2DCoreOp : ModelCoreOp<"conv2d_core">;
def LinearCoreOp : ModelCoreOp<"linear_core">;
def QCoreOp : ModelCoreOp<"q_core">;
def KCoreOp : ModelCoreOp<"k_core">;
def VCoreOp : ModelCoreOp<"v_core">;
def ZCoreOp : ModelCoreOp<"z_core">;
def FCCoreOp : ModelCoreOp<"fc_core">;
def AffineCoreOp : ModelCoreOp<"affine_core">;
def NormCoreOp : ModelCoreOp<"norm_core">;
def QKCoreOp : ModelCoreOp<"qk_core">;
def QKVCoreOp : ModelCoreOp<"qkv_core">;
def ResidualCoreOp : ModelCoreOp<"residual_core">;
def RescaleCoreOp : ModelCoreOp<"rescale_core">;
def PoolCoreOp : ModelCoreOp<"pool_core">;

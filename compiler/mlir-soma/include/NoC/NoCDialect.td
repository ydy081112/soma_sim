include "mlir/IR/OpBase.td"

def NoCDialect : Dialect {
  let name = "noc";
  let cppNamespace = "::noc";
  let summary = "core-level 片上网络资源声明";
  let description = [{仅描述 network 的拓扑、尺寸、单 hop 时延与路由策略。}];
  let useDefaultAttributePrinterParser = 0;
  let extraClassDeclaration = [{
    ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type type) const override;
    void printAttribute(::mlir::Attribute attribute,
                        ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

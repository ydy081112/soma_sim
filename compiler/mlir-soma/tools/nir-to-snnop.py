#!/usr/bin/env python3
"""将 NIR 1.x HDF5 block graph 导入单个硬件无关的 snn_op MLIR 模块。"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import h5py

def text(value): return value.decode() if isinstance(value, bytes) else str(value)
def scalar(ds):
    value = ds[()]
    return value.reshape(-1)[0].item() if getattr(value, "shape", ()) else value.item() if hasattr(value, "item") else value
def ttype(shape, element): return "tensor<" + "x".join(map(str, shape)) + "x" + element + ">"
def mlir_scalar_type(dtype):
    """NIR HDF5 dtype 到 MLIR 标量 type；MLIR 整数按惯例使用 signless iN。"""
    kind, bits = dtype.kind, dtype.itemsize * 8
    if kind in ("i", "u"): return f"i{bits}"
    if kind == "f" and bits in (16, 32, 64): return f"f{bits}"
    if kind == "b": return "i1"
    raise ValueError(f"unsupported NIR dtype: {dtype}")
def output_shape(group):
    data = group.get("metadata/output")
    if data is None: raise ValueError(f"{group.name}: missing metadata/output")
    return tuple(map(int, data.shape))
def input_shape(group):
    data = group.get("metadata/input")
    if isinstance(data, h5py.Dataset): return tuple(map(int, data.shape))
    return output_shape(group)
def output_type(group):
    data = group.get("metadata/output")
    if data is None: raise ValueError(f"{group.name}: missing metadata/output")
    return ttype(output_shape(group), mlir_scalar_type(data.dtype))
def scalar_attr(dataset, default_value=None, default_dtype=None):
    """保留 NIR dataset 的原生标量 dtype，而不是统一转成 f64。"""
    if dataset is None:
        if default_dtype is None: raise ValueError("missing dtype for default scalar attribute")
        value, dtype = default_value, default_dtype
    else:
        value, dtype = scalar(dataset), dataset.dtype
    mlir_type = mlir_scalar_type(dtype)
    if dtype.kind == "f":
        literal = repr(float(value))
        if "." not in literal and "e" not in literal.lower(): literal += ".0"
    else:
        literal = str(int(value))
    return f"{literal} : {mlir_type}"
def attr_string(value): return json.dumps(str(value))

def symbol_fragment(value): return re.sub(r"[^A-Za-z0-9_]", "_", value)
def param_symbol(prefix, node, kind): return f"{prefix}_{symbol_fragment(node)}_{kind}"
def param_line(symbol, source, kind, dataset):
    shape = ", ".join(str(int(v)) for v in dataset.shape)
    return f"  snn_op.param @{symbol} {{dtype = {attr_string(dataset.dtype)}, kind = {attr_string(kind)}, shape = array<i64: {shape}>, source = {attr_string(source)}}}"

LINEAR_NODES = {"q_IF", "k_IF", "v_IF", "proj_IF", "fc1_IF", "fc2_IF"}
LINEAR_OPS = {"q_IF": "x_wq", "k_IF": "x_wk", "v_IF": "x_wv",
              "proj_IF": "z_wo", "fc1_IF": "fc", "fc2_IF": "fc"}
AFFINE_NODES = {"norm1_IF", "norm1_IF_next", "norm2_IF"}
RESIDUAL_NODES = {"residual_add1_IF", "residual_add2_IF"}
QK_NODE = "qk_multi_IF"
AV_NODE = "qkv_multi_IF"
BLOCK_INPUT_NODE = "block_input"
SUPPORTED_NODES = (LINEAR_NODES | AFFINE_NODES | RESIDUAL_NODES |
                   {QK_NODE, AV_NODE, BLOCK_INPUT_NODE})
NODE_ORDER = [BLOCK_INPUT_NODE, "q_IF", "k_IF", "v_IF", QK_NODE, AV_NODE,
              "proj_IF", "norm1_IF", "residual_add1_IF", "norm1_IF_next",
              "fc1_IF", "fc2_IF", "residual_add2_IF", "norm2_IF"]
NODE_ORDER_INDEX = {name: index for index, name in enumerate(NODE_ORDER)}

def node_order_key(name): return (NODE_ORDER_INDEX.get(name, len(NODE_ORDER)), name)

def render_block(source: Path, function_name: str, prefix: str):
    """返回 module body；prefix 保证归并时 symbol 全局唯一。"""
    with h5py.File(source) as f:
        if text(scalar(f["version"])) != "1.0.8": raise ValueError("only NIR 1.0.8 is supported in V1")
        nodes = f["node/nodes"]
        edges = [(text(a), text(b)) for a, b in f["node/edges"][...]]
        names = list(nodes.keys())
        incoming = {n: [] for n in names}
        for a, b in edges:
            if a in incoming and b in incoming: incoming[b].append(a)
        # NIR archive order is not execution order; stable Kahn traversal is deterministic.
        ordered, ready = [], sorted((n for n in names if not incoming[n]), key=node_order_key)
        remaining = {n: list(v) for n, v in incoming.items()}
        while ready:
            n = ready.pop(0); ordered.append(n)
            for a, b in edges:
                if a == n and b in remaining and n in remaining[b]:
                    remaining[b].remove(n)
                    if not remaining[b]: ready.append(b); ready.sort(key=node_order_key)
        if len(ordered) != len(names): raise ValueError("NIR graph contains a cycle or unknown edge")
        sources = [n for n in ordered if not incoming[n]]
        args = []
        for n in sources:
            args.append(f"%arg_{n}: {ttype(input_shape(nodes[n]), '!snn.spike<ternary>')}")
        params = []
        parameter_symbols = {}
        for n in ordered:
            for kind in ("weight", "bias"):
                dataset = nodes[n].get(f"metadata/{kind}")
                if dataset is None: continue
                symbol = param_symbol(prefix, n, kind)
                parameter_symbols[(n, kind)] = symbol
                uri = f"nir://{source.name}#/node/nodes/{n}/metadata/{kind}"
                params.append(param_line(symbol, uri, kind, dataset))
        outgoing = {n for n in names for a, _ in edges if a == n}
        terminals = [n for n in ordered if n != BLOCK_INPUT_NODE and n not in outgoing]
        if len(terminals) != 1:
            raise ValueError(f"expected exactly one computational terminal, got: {', '.join(terminals)}")
        terminal = terminals[0]
        terminal_type = output_type(nodes[terminal])
        lines = [f"  func.func @{function_name}({', '.join(args)}) -> {terminal_type} {{"]
        unknown = set(names) - SUPPORTED_NODES
        if unknown:
            raise ValueError(f"unsupported NIR node name(s): {', '.join(sorted(unknown))}")

        values = {}
        def parent_type(parent):
            return (ttype(output_shape(nodes[parent]), "!snn.spike<ternary>")
                    if parent == BLOCK_INPUT_NODE else output_type(nodes[parent]))
        for n in ordered:
            group = nodes[n]
            accum = output_type(group)
            parents = incoming[n]
            time_dim = 0
            value = f"%{n}"
            if n == BLOCK_INPUT_NODE:
                if parents:
                    raise ValueError(f"{n}: block_input must not have parents")
                values[n] = f"%arg_{n}"
                continue
            if n == QK_NODE:
                if len(parents) != 2:
                    raise ValueError(f"{n}: qk requires exactly two parents")
                q, k = (values[p] for p in parents)
                head_dim = output_shape(nodes[parents[0]])[-1]
                lines.append(f"    %{n}_acc = snn_op.qk {q}, {k} {{head_dim = {head_dim} : i64, num_heads = {output_shape(nodes[parents[0]])[-3]} : i64, scale = 1.0 : f64, time_dim = 0 : i64}} : ({parent_type(parents[0])}, {parent_type(parents[1])}) -> {accum}")
            elif n == AV_NODE:
                if len(parents) != 2:
                    raise ValueError(f"{n}: av requires exactly two parents")
                a, v = (values[p] for p in parents); head_dim = output_shape(group)[-1]
                lines.append(f"    %{n}_acc = snn_op.qkv {a}, {v} {{head_dim = {head_dim} : i64, num_heads = {output_shape(group)[-3]} : i64, time_dim = 0 : i64}} : ({parent_type(parents[0])}, {parent_type(parents[1])}) -> {accum}")
            elif n in RESIDUAL_NODES:
                if len(parents) != 2:
                    raise ValueError(f"{n}: residual requires exactly two parents")
                a, b = (values[p] for p in parents)
                shift_main = int(scalar(group["metadata/shifter1"]))
                shift_skip = int(scalar(group["metadata/shifter2"]))
                lines.append(f"    %{n}_acc = snn_op.residual {a}, {b} {{time_dim = 0 : i64, w_main = {1 << shift_main} : i64, w_skip = {1 << shift_skip} : i64}} : ({parent_type(parents[0])}, {parent_type(parents[1])}) -> {accum}")
            elif n in LINEAR_NODES:
                if len(parents) != 1:
                    raise ValueError(f"{n}: linear requires exactly one parent")
                if (n, "weight") not in parameter_symbols:
                    raise ValueError(f"{n}: linear requires metadata/weight")
                parent = values[parents[0]]
                inp = parent_type(parents[0])
                weight = group["metadata/weight"]
                attributes = [f"in_features = {weight.shape[1]} : i64", f"out_features = {weight.shape[0]} : i64", f"time_dim = {time_dim} : i64", f"weight = @{parameter_symbols[(n, 'weight')]}" ]
                if (n, "bias") in parameter_symbols: attributes.append(f"bias = @{parameter_symbols[(n, 'bias')]}")
                lines.append(f"    %{n}_acc = snn_op.{LINEAR_OPS[n]} {parent} {{{', '.join(attributes)}}} : ({inp}) -> {accum}")
            elif n in AFFINE_NODES:
                if len(parents) != 1:
                    raise ValueError(f"{n}: affine requires exactly one parent")
                if (n, "weight") not in parameter_symbols or (n, "bias") not in parameter_symbols:
                    raise ValueError(f"{n}: affine requires metadata/weight and metadata/bias")
                parent = values[parents[0]]
                inp = parent_type(parents[0])
                lines.append(f"    %{n}_acc = snn_op.norm {parent} {{axis = -1 : i64, bias = @{parameter_symbols[(n, 'bias')]}, time_dim = {time_dim} : i64, weight = @{parameter_symbols[(n, 'weight')]}}} : ({inp}) -> {accum}")
            else:
                raise ValueError(f"unsupported NIR node name: {n}")
            # 旧 NIR 使用 node/threshold；当前 split-residual-norm NIR 将同一
            # ST-BIF 阈值放在 metadata/multiplier。两者都沿用原有的标量 ST-BIF
            # 表达（数组时取第一个通道），不改 tracer 的处理方式。
            threshold_data = group.get("threshold")
            if threshold_data is None: threshold_data = group.get("metadata/multiplier")
            if threshold_data is None:
                raise ValueError(f"{n}: missing threshold or metadata/multiplier")
            output_dataset = group["metadata/output"]
            threshold = scalar_attr(threshold_data)
            tr_min = scalar_attr(group.get("metadata/tracer_min"), 0, output_dataset.dtype)
            tr_max = scalar_attr(group.get("metadata/tracer_max"), 1, output_dataset.dtype)
            voltage_dtype = mlir_scalar_type(threshold_data.dtype)
            lines.append(f"    %{n}_stbif = snn_op.st_bif %{n}_acc {{threshold = {threshold}, time_dim = 0 : i64, tr_max = {tr_max}, tr_min = {tr_min}, voltage_dtype = \"{voltage_dtype}\"}} : {accum} -> {accum}")
            values[n] = f"%{n}_stbif"
        lines.extend([f"    return {values[terminal]} : {terminal_type}", "  }"])
    return params, lines

def convert(source: Path, destination: Path):
    # NIR 的 HDF5 文件在部分文件系统上也会被识别为目录，故以扩展名判定。
    blocks = sorted(source.glob("*.nir")) if source.is_dir and source.suffix != ".nir" else [source]
    if not blocks: raise ValueError(f"{source}: contains no .nir blocks")
    params, functions = [], []
    for index, block in enumerate(blocks):
        stem = symbol_fragment(block.stem)
        block_params, block_function = render_block(block, "block" if len(blocks) == 1 else f"block_{index:02d}", f"b{index:02d}_{stem}")
        params.extend(block_params); functions.extend(block_function)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("module attributes {snn_op.frontend = \"nir-1.0.8\", snn_op.weight_source = \"nir\"} {\n" + "\n".join(params + functions) + "\n}\n", encoding="utf-8")

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); convert(args.input, args.output)

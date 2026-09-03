#!/usr/bin/env python3
"""将 NIR 1.x HDF5 block graph 导入单个硬件无关的 snn_op MLIR 模块。"""
from __future__ import annotations
import argparse
import json
import re
import tempfile
from pathlib import Path
import h5py

def text(value): return value.decode() if isinstance(value, bytes) else str(value)
def scalar(ds):
    value = ds[()]
    return value.reshape(-1)[0].item() if getattr(value, "shape", ()) else value.item() if hasattr(value, "item") else value
def ttype(shape, element): return "tensor<" + "x".join(map(str, shape)) + "x" + element + ">"
def output_shape(group):
    data = group.get("metadata/output")
    if data is None: raise ValueError(f"{group.name}: missing metadata/output")
    return tuple(map(int, data.shape))
def input_shape(group):
    data = group.get("metadata/input")
    if isinstance(data, h5py.Dataset): return tuple(map(int, data.shape))
    return output_shape(group)
def attr_string(value): return json.dumps(str(value))

def convert(source: Path, destination: Path):
    # NIR 的 HDF5 文件在部分文件系统上也会被识别为目录，故以扩展名判定。
    if source.is_dir and source.suffix != ".nir":
        blocks = sorted(source.glob("*.nir"))
        if not blocks: raise ValueError(f"{source}: contains no .nir blocks")
        with tempfile.TemporaryDirectory() as temporary:
            bodies = []
            for index, block in enumerate(blocks):
                part = Path(temporary) / f"{index}.mlir"
                convert(block, part)
                lines = part.read_text(encoding="utf-8").splitlines()
                body = lines[1:-1]
                body = [re.sub(r"func.func @block\(", f"func.func @block_{index:02d}(", line) for line in body]
                bodies.extend(body)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("module attributes {snn_op.frontend = \"nir-1.0.8\", snn_op.weight_source = \"nir\"} {\n" + "\n".join(bodies) + "\n}\n", encoding="utf-8")
        return
    with h5py.File(source) as f:
        if text(scalar(f["version"])) != "1.0.8": raise ValueError("only NIR 1.0.8 is supported in V1")
        nodes = f["node/nodes"]
        edges = [(text(a), text(b)) for a, b in f["node/edges"][...]]
        names = list(nodes.keys())
        incoming = {n: [] for n in names}
        for a, b in edges:
            if a in incoming and b in incoming: incoming[b].append(a)
        # NIR archive order is not execution order; stable Kahn traversal is deterministic.
        ordered, ready = [], sorted(n for n in names if not incoming[n])
        remaining = {n: list(v) for n, v in incoming.items()}
        while ready:
            n = ready.pop(0); ordered.append(n)
            for a, b in edges:
                if a == n and b in remaining and n in remaining[b]:
                    remaining[b].remove(n)
                    if not remaining[b]: ready.append(b); ready.sort()
        if len(ordered) != len(names): raise ValueError("NIR graph contains a cycle or unknown edge")
        sources = [n for n in ordered if not incoming[n]]
        args = []
        for n in sources:
            args.append(f"%arg_{n}: {ttype(input_shape(nodes[n]), '!snn.spike<ternary>')}")
        lines = ["module attributes {snn_op.frontend = \"nir-1.0.8\", snn_op.weight_source = \"nir\"} {",
                 f"  func.func @block({', '.join(args)}) {{"]
        values = {}
        for n in ordered:
            group = nodes[n]; out = ttype(output_shape(group), "!snn.spike<ternary>")
            accum = ttype(output_shape(group), "i32")
            parents = incoming[n]
            threshold = float(scalar(group["threshold"]))
            tr_min = float(scalar(group.get("metadata/tracer_min", [0])))
            tr_max = float(scalar(group.get("metadata/tracer_max", [1])))
            time_dim = 0
            value = f"%{n}"
            if n == "qk_multi_IF" and len(parents) == 2:
                q, k = (values[p] for p in parents)
                head_dim = output_shape(nodes[parents[0]])[-1]
                lines.append(f"    %{n}_acc = snn_op.qk {q}, {k} {{head_dim = {head_dim} : i64, num_heads = {output_shape(nodes[parents[0]])[-3]} : i64, scale = 1.0 : f64, time_dim = 0 : i64}} : ({q and ttype(output_shape(nodes[parents[0]]), '!snn.spike<ternary>')}, {ttype(output_shape(nodes[parents[1]]), '!snn.spike<ternary>')}) -> {accum}")
            elif n == "qkv_multi_IF" and len(parents) == 2:
                a, v = (values[p] for p in parents); head_dim = output_shape(group)[-1]
                lines.append(f"    %{n}_acc = snn_op.av {a}, {v} {{head_dim = {head_dim} : i64, num_heads = {output_shape(group)[-3]} : i64, time_dim = 0 : i64}} : ({ttype(output_shape(nodes[parents[0]]), '!snn.spike<ternary>')}, {ttype(output_shape(nodes[parents[1]]), '!snn.spike<ternary>')}) -> {accum}")
            elif len(parents) == 2:
                a, b = (values[p] for p in parents)
                lines.append(f"    %{n}_acc = snn_op.residual {a}, {b} {{time_dim = 0 : i64}} : ({ttype(output_shape(nodes[parents[0]]), '!snn.spike<ternary>')}, {ttype(output_shape(nodes[parents[1]]), '!snn.spike<ternary>')}) -> {accum}")
            else:
                parent = values[parents[0]] if parents else f"%arg_{n}"
                inp = ttype(output_shape(nodes[parents[0]]), "!snn.spike<ternary>") if parents else ttype(input_shape(group), "!snn.spike<ternary>")
                weight = group.get("metadata/weight")
                if weight is not None and weight.ndim == 2:
                    lines.append(f"    %{n}_acc = snn_op.linear {parent} {{bias = {attr_string('nir://' + source.name + '#/node/nodes/' + n + '/metadata/bias')}, in_features = {weight.shape[1]} : i64, out_features = {weight.shape[0]} : i64, spike_encoding = \"ternary\", time_dim = {time_dim} : i64, weight = {attr_string('nir://' + source.name + '#/node/nodes/' + n + '/metadata/weight')}}} : ({inp}) -> {accum}")
                else:
                    lines.append(f"    %{n}_acc = snn_op.rescale {parent} {{scale = 1.0 : f64, time_dim = {time_dim} : i64}} : ({inp}) -> {accum}")
            lines.append(f"    {value} = snn_op.st_bif %{n}_acc {{neuron_model = \"st_bif\", threshold = {threshold} : f64, time_dim = 0 : i64, tr_max = {tr_max} : f64, tr_min = {tr_min} : f64}} : ({accum}) -> {out}")
            values[n] = value
        lines.extend(["    return", "  }", "}", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); convert(args.input, args.output)

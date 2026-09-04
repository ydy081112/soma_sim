#!/usr/bin/env python3
"""只读 NIR，导出 SOMA ViT runtime 资产；所有算子按 metadata/graph 判定。"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import h5py
import numpy as np


def scalar(v):
    v = v[()]
    if getattr(v, "shape", ()): v = v.reshape(-1)[0]
    return v.item() if hasattr(v, "item") else v


def topo(names, edges):
    inc = {n: [] for n in names}
    for a, b in edges: inc[b].append(a)
    ready, order = sorted(n for n in names if not inc[n]), []
    while ready:
        n = ready.pop(0); order.append(n)
        for a, b in edges:
            if a == n and a in inc[b]:
                inc[b].remove(a)
                if not inc[b]: ready.append(b); ready.sort()
    if len(order) != len(names): raise ValueError("NIR graph 不是 DAG")
    return order


def shifted(values, shifts):
    values = np.asarray(values, dtype=np.int64)
    shifts = np.broadcast_to(np.asarray(shifts, dtype=np.int64).reshape(-1), values.shape)
    return np.array([v << int(s) if s >= 0 else v >> -int(s)
                     for v, s in zip(values.flat, shifts.flat)], dtype=np.int64).reshape(values.shape)


def emit_item(lines, indent, values):
    def fmt(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, list): return "[" + ", ".join(map(str, v)) + "]"
        return str(v)
    first, *rest = values.items(); p = " " * indent
    lines.append(f"{p}- {first[0]}: {fmt(first[1])}")
    for k, v in rest: lines.append(f"{p}  {k}: {fmt(v)}")


def layer_channels(shape, weight_ndim):
    if len(shape) == 5 and weight_ndim == 2: return int(shape[-3] * shape[-1])
    if len(shape) == 5 and weight_ndim is None and shape[-1] != shape[-2]: return int(shape[-3] * shape[-1])
    return int(shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nir", type=Path, required=True); ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--input-csv", type=Path, required=True); ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True); ap.add_argument("--input-tensor", required=True)
    ap.add_argument("--max-neurons", type=int, default=1024); args = ap.parse_args()
    if "/metadata/" not in args.input_tensor: raise ValueError("input-tensor 必须是 node/metadata/dataset")
    input_node, input_key = args.input_tensor.split("/metadata/", 1)
    arrays, nodes, meta = {}, {}, {}
    with h5py.File(args.nir, "r") as h5:
        version = scalar(h5["version"]); version = version.decode() if isinstance(version, bytes) else str(version)
        if version != "1.0.8": raise ValueError("only NIR 1.0.8 is supported")
        graph = h5["node/nodes"]; edges = [[a.decode(), b.decode()] for a, b in h5["node/edges"][...]]
        if input_node not in graph: raise ValueError("input-tensor node 不存在")
        order, shapes = topo(list(graph), edges), {}
        for name, group in graph.items():
            md, entry = group["metadata"], {"type": str(scalar(group["type"])), "datasets": {}}
            for key, value in md.items():
                if isinstance(value, h5py.Dataset):
                    arrays[f"{name}__{key}"] = value[...]; entry["datasets"][key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                else:
                    entry["datasets"][key] = {s: {"shape": list(ds.shape), "dtype": str(ds.dtype)} for s, ds in value.items()}
                    for s, ds in value.items(): arrays[f"{name}__{key}__{s}"] = ds[...]
            nodes[name] = entry; shapes[name] = tuple(md["output"].shape)
            if name == input_node: continue
            if not all(k in md for k in ("multiplier", "tracer_min", "tracer_max")): raise ValueError(f"{name}: 缺少 ST-BIF metadata")
            threshold = np.asarray(md["multiplier"], dtype=np.int64).reshape(-1); arrays[f"{name}_threshold"] = threshold.astype(np.float32)
            bias = np.asarray(md.get("bias", np.zeros(1, dtype=np.int64)), dtype=np.int64).reshape(-1)
            replay = scalar(group["type"]) in (b"Input", "Input")
            if bias.size != 1 or int(bias[0]) != 0:
                if not replay: bias = shifted(bias, md["shifter"])
                init_threshold = np.full(bias.shape, threshold[0]) if threshold.size == 1 else threshold
                initial = init_threshold // 2 + bias
            else: initial = threshold // 2
            arrays[f"{name}_initial_membrane"] = initial.astype(np.float32)
            weight = md.get("weight"); ndim = None if weight is None else weight.ndim
            meta[name] = {"weight_ndim": ndim, "threshold": threshold, "tracer_min": int(scalar(md["tracer_min"])),
                          "tracer_max": int(scalar(md["tracer_max"])), "residual": "input1" in md and "input2" in md,
                          "qkv": isinstance(md.get("input"), h5py.Group)}
            if weight is not None:
                raw = np.asarray(weight, dtype=np.int64)
                if ndim == 2:
                    shift = np.asarray(md["shifter"], dtype=np.int64).reshape(-1)
                    if shift.size != 1: raise ValueError(f"{name}: projection shifter 必须为 scalar")
                    # NIR shifter 是乘以 2**shift；负值保留为分数权重，不能对
                    # 单个 signed weight 作整数右移，否则会改变逐 timestep ST-BIF 状态。
                    scaled = shifted(raw, shift[0]) if shift[0] >= 0 else raw.astype(np.float64) * (2.0 ** int(shift[0]))
                    arrays[f"{name}_weight"] = scaled.T.astype(np.float32)
                elif ndim == 1: arrays[f"{name}_weight"] = shifted(raw, md["shifter"]).astype(np.float32)
                else: raise ValueError(f"{name}: 不支持 weight rank")
        x = np.asarray(graph[input_node]["metadata"][input_key])
    if x.ndim != 4 or x.shape[1] != 1: raise ValueError("input tensor 必须为 [T,1,token,embedding]")
    tcount, _, tokens, embedding = map(int, x.shape); arrays["input_shape"] = np.asarray(x.shape, dtype=np.int64)
    nonzero_coordinates = np.argwhere(x[:, 0] != 0)
    last_input_timestep = int(nonzero_coordinates[:, 0].max()) + 1 if len(nonzero_coordinates) else 0
    args.input_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.input_csv.open("w", newline="", encoding="utf-8") as out:
        w = csv.writer(out); w.writerow(["generated_time", "current_time", "spike_id", "timestep", "layer_id", "src_neuron", "src_pe", "src_router", "dst_pe", "dst_router", "value"])
        for sid, (t, token, chan) in enumerate(nonzero_coordinates):
            w.writerow([0, 0, sid, int(t) + 1, "input", int(token) * embedding + int(chan), 0, 0, 0, 0, int(x[t, 0, token, chan])])

    # selected NIR source is represented by a single virtual SOMA input layer.
    edges = [["input" if a == input_node else a, b] for a, b in edges if b != input_node]
    compute = [n for n in order if n != input_node]; incoming = {n: [a for a, b in edges if b == n] for n in compute}
    depth = {"input": 0}
    for n in compute: depth[n] = max(depth[a] for a in incoming[n]) + 1
    sizes = {"input": tokens * embedding, **{n: int(np.prod(shapes[n][2:])) for n in compute}}
    starts, layers, start = {}, [], 0
    for n in ["input"] + sorted(compute, key=lambda q: (depth[q], q)):
        count = sizes[n]; cores = (count + args.max_neurons - 1) // args.max_neurons; starts[n] = start
        if n == "input": layer = {"id": n, "op": "input", "pe": 0, "core": 0, "router": 0, "neurons": count, "output_channels": embedding, "virtual_input": True}
        else:
            d = meta[n]; layer = {"id": n, "op": "linear", "pe": start // 4, "core": start % 4, "router": start // 4,
                "neurons": count, "output_channels": layer_channels(shapes[n], d["weight_ndim"]), "source_neurons": sizes["input"], "weight_prefix": n, "state_start_timestep": depth[n] + 1,
                "neuron_model": "st_bif", "leak": 1, "threshold": float(d["threshold"][0]), "tracer_min": d["tracer_min"], "tracer_max": d["tracer_max"], "aggregate_core_count": cores}
            if len(incoming[n]) == 2 and not d["residual"]:
                kind = "qkv" if d["qkv"] else "qk"; lhs, rhs = shapes[incoming[n][0]], shapes[incoming[n][1]]
                shifts = np.asarray(arrays[f"{n}__shifter"], dtype=np.int64).reshape(-1)
                if shifts.size != 2 or shifts[0] != shifts[1]: raise ValueError(f"{n}: attention shifter 不一致")
                layer.update({"operator_type": "incremental_spike_matmul", "attention_kind": kind, "attention_heads": int(lhs[-3]), "attention_rows": int(lhs[-2]), "attention_reduction": int(lhs[-1]), "attention_columns": int(rhs[-1]) if kind == "qkv" else int(rhs[-2]), "attention_output_layout": "row_head_column" if kind == "qkv" else "head_row_column", "attention_accumulation_scale": 1 << int(shifts[0])})
                if kind == "qkv":
                    # QKV 的 NIR [H,R,C] 采用 runtime [R,H*C]；R 与 C 相等时也不能靠 shape 猜测。
                    layer["output_channels"] = int(lhs[-3]) * int(rhs[-1])
            elif d["weight_ndim"] == 2:
                projection_shift = int(np.asarray(arrays[f"{n}__shifter"], dtype=np.int64).reshape(-1)[0])
                if projection_shift < 0:
                    layer["post_accumulation_rounding"] = "nearest_even"
            # ViT block correctness needs the final ST-BIF signed trace; final_scores() already
            # falls back to the terminal layer, so do not suppress its firing as a readout.
        layers.append(layer); start += cores
    conns, residual = [], {}
    for source, target in edges:
        d, parents = meta[target], incoming[target]; delay = max(depth[p] for p in parents) - depth[source]
        if d["residual"]:
            operand = "input1" if source == parents[0] else "input2"; idx = 1 if operand == "input1" else 2
            shift = int(np.asarray(arrays[f"{target}__shifter{idx}"], dtype=np.int64).reshape(-1)[0]); prefix = f"{target}__{operand}"
            arrays[f"{prefix}_weight"] = np.array([1 << shift], dtype=np.float32)
            conns.append({"from": source, "to": target, "type": "identity", "hardware_type": "identity", "weight_prefix": prefix, "operand": operand, "delay": delay})
            residual.setdefault(target, {"operands": {}})["operands"][operand] = {"source": source, "shift": shift, "weight": 1 << shift, "delay": delay}
        elif len(parents) == 2:
            kind = "qkv" if d["qkv"] else "qk"
            if kind == "qk": layout = "row_head_reduction"
            else: layout = "head_row_reduction" if source == parents[0] else "row_head_column"
            conns.append({"from": source, "to": target, "type": "attention_operand", "hardware_type": "dense", "weight_prefix": target, "operand": "lhs" if source == parents[0] else "rhs", "operand_layout": layout, "delay": delay})
        elif d["weight_ndim"] == 1: conns.append({"from": source, "to": target, "type": "identity", "hardware_type": "identity", "weight_prefix": target, "delay": delay})
        else: conns.append({"from": source, "to": target, "type": "grouped_dense", "hardware_type": "dense", "weight_prefix": target, "delay": delay})
    for target, rec in residual.items():
        for operand, value in rec["operands"].items():
            source = input_node if value["source"] == "input" else value["source"]
            source_out = x if source == input_node else arrays[f"{source}__output"]
            if not np.array_equal(arrays[f"{target}__{operand}"], source_out): raise ValueError(f"{target}/{operand} 与父节点 output 不一致")
    args.npz.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.npz, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
    args.manifest.parent.mkdir(parents=True, exist_ok=True); args.manifest.write_text(json.dumps({"nir": str(args.nir), "input_tensor": args.input_tensor, "input_layout": "[T,B,token,embedding]", "input_shape": list(x.shape), "virtual_input_neurons": tokens * embedding, "edges": edges, "nodes": nodes, "residual": residual}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Input CSV omits zero events, but Cx State must keep evolving through every recorded NIR frame.
    flush = tcount + max(depth.values()) - last_input_timestep
    lines = ["mapping:", "  model: vit_block00_nir_derived", f"  flush_timesteps: {flush}", "  signed_firing_trace: true", "  layers:"]
    for v in layers: emit_item(lines, 4, v)
    lines.append("  connections:"); [emit_item(lines, 4, v) for v in conns]
    lines.append("  routes:"); [emit_item(lines, 4, {"from": c["from"], "to": c["to"], "routers": [starts[c["from"]] // 4, starts[c["to"]] // 4]}) for c in conns]
    args.mapping.parent.mkdir(parents=True, exist_ok=True); args.mapping.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()

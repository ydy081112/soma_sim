#!/usr/bin/env python3
"""逐 timestep 对照 SOMA signed firing trace 与 NIR 所有计算节点 output。"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np


def flatten_reference(name, raw, node, parent_count):
    # Runtime projection/QKV 使用 token-major，NIR 的五维 tensor 是 head-major。
    if raw.ndim == 3: return raw.reshape(raw.shape[0], -1)
    if raw.ndim != 4: raise ValueError(f"{name}: 不支持 output rank {raw.ndim + 1}")
    datasets = node["datasets"]
    if parent_count == 2 and "input1" in datasets: return raw.reshape(raw.shape[0], -1)
    if parent_count == 2 and isinstance(datasets.get("input"), dict) and "shape" not in datasets["input"]:
        return raw.transpose(0, 2, 1, 3).reshape(raw.shape[0], -1)  # QKV [H,R,C] -> [R,H,C]
    if parent_count == 2: return raw.reshape(raw.shape[0], -1)      # QK [H,R,R]
    if datasets.get("weight", {}).get("shape") and len(datasets["weight"]["shape"]) == 2:
        return raw.transpose(0, 2, 1, 3).reshape(raw.shape[0], -1)  # projection [H,R,C] -> [R,H,C]
    return raw.reshape(raw.shape[0], -1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True); ap.add_argument("--markdown", type=Path, required=True)
    args = ap.parse_args(); manifest = json.loads(args.manifest.read_text(encoding="utf-8")); archive = np.load(args.npz)
    input_node = manifest["input_tensor"].split("/metadata/", 1)[0]
    names = [n for n in manifest["nodes"] if n != input_node]
    incoming = {n: [] for n in names}
    for source, target in manifest["edges"]:
        if target in incoming: incoming[target].append(source)
    depth = {"input": 0}; remaining = set(names)
    while remaining:
        ready = sorted(n for n in remaining if all(p in depth for p in incoming[n]))
        if not ready: raise ValueError("mapping graph 不是 DAG 或缺少外部输入")
        for n in ready: depth[n] = max(depth[p] for p in incoming[n]) + 1; remaining.remove(n)
    trace = {}
    with args.trace.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream): trace[(row["layer_id"], int(row["timestep"]), int(row["neuron"]))] = int(float(row["value"]))
    results = []
    for name in sorted(names, key=lambda n: (depth[n], n)):
        reference = flatten_reference(name, archive[f"{name}__output"][:, 0], manifest["nodes"][name], len(incoming[name]))
        actual = np.zeros_like(reference)
        for (layer, sim_t, neuron), value in trace.items():
            t = sim_t - 1 - depth[name]
            if layer == name and 0 <= t < actual.shape[0] and neuron < actual.shape[1]: actual[t, neuron] = value
        mismatch = int(np.count_nonzero(actual != reference)); first = None
        if mismatch:
            index = tuple(map(int, np.argwhere(actual != reference)[0])); first = {"index": list(index), "soma": int(actual[index]), "nir": int(reference[index])}
        results.append({"layer": name, "mismatches": mismatch, "elements": int(reference.size), "soma_nonzero": int(np.count_nonzero(actual)), "nir_nonzero": int(np.count_nonzero(reference)), "first_mismatch": first})
    report = {"all_computational_layers_exact": all(x["mismatches"] == 0 for x in results), "layers": results,
              "incremental_identity_test": "soma-focused-tests 覆盖 QK/QKV 的 +1/0/-1 输入"}
    args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# ViT block00 全层正确性对比", "", f"所有计算节点逐 timestep、逐元素完全一致：{'是' if report['all_computational_layers_exact'] else '否'}。", "", "| 层 | mismatch / elements | SOMA 非零 | NIR 非零 |", "|---|---:|---:|---:|"]
    lines += [f"| {x['layer']} | {x['mismatches']} / {x['elements']} | {x['soma_nonzero']} | {x['nir_nonzero']} |" for x in results]
    lines += ["", "QK/QKV 另由 `soma-focused-tests` 以 +1/0/-1 输入逐 timestep 验证增量恒等式。"]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__": main()

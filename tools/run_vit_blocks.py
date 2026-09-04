#!/usr/bin/env python3
"""按配置连续回放多个 ViT NIR block，并汇总端到端统计。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml


def run(command: list[object]) -> None:
    print("运行：", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def terminal_node(h5: h5py.File) -> str:
    names = list(h5["node/nodes"])
    edges = [(a.decode(), b.decode()) for a, b in h5["node/edges"][...]]
    outgoing = {name: 0 for name in names}
    for source, _ in edges:
        outgoing[source] += 1
    leaves = [name for name in names if outgoing[name] == 0]
    if len(leaves) != 1:
        raise ValueError(f"NIR 必须只有一个终端输出，实际为 {leaves}")
    return leaves[0]


def boundary_equal(previous: Path, current: Path, input_tensor: str) -> dict[str, object]:
    input_node, input_key = input_tensor.split("/metadata/", 1)
    with h5py.File(previous, "r") as old_h5, h5py.File(current, "r") as new_h5:
        target = np.asarray(new_h5[f"node/nodes/{input_node}/metadata/{input_key}"])
        candidates = []
        for name, group in old_h5["node/nodes"].items():
            dataset = group["metadata"].get("output")
            if dataset is not None and dataset.shape == target.shape and np.array_equal(dataset[...], target):
                candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(f"无法唯一定位 {previous.name} 到 {current.name} 的边界产出：{candidates}")
    return {
        "from_nir": str(previous), "from_output": candidates[0],
        "to_nir": str(current), "to_input": input_tensor,
        "shape": list(target.shape), "mismatch_elements": 0,
        "exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    vit = config["vit_blocks"]
    if vit["mapping_strategy"] != "adjacent_qkv_attention_then_projection_mlp":
        raise ValueError("当前工具不支持所选 mapping_strategy")
    blocks = [int(block) for block in vit["blocks"]]
    if not blocks or blocks != sorted(set(blocks)):
        raise ValueError("blocks 必须是非空、严格递增且不重复的整数列表")
    nir_dir = Path(vit["nir_directory"])
    nirs = [nir_dir / vit["nir_pattern"].format(block=block) for block in blocks]
    missing = [str(path) for path in nirs if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到 NIR：" + ", ".join(missing))

    boundaries = [boundary_equal(a, b, vit["input_tensor"]) for a, b in zip(nirs, nirs[1:])]
    if not all(item["exact"] for item in boundaries):
        raise ValueError("NIR block 边界不连续，拒绝把它们汇总为端到端结果")

    runtime_root = root / vit["runtime_root"]
    mapping_root = root / vit["mapping_root"]
    output_root = root / vit["output_root"]
    records: list[dict[str, object]] = []
    for block, nir in zip(blocks, nirs):
        tag = f"block{block:02d}"
        npz = runtime_root / f"{tag}_runtime.npz"
        spikes = runtime_root / f"{tag}_input_spike.csv"
        manifest = mapping_root / f"{tag}_nir_manifest.json"
        mapping = mapping_root / f"{tag}_mapping.yaml"
        output = output_root / tag
        run([sys.executable, root / "tools/prepare_vit_block00_assets.py",
             "--nir", nir, "--npz", npz, "--input-csv", spikes,
             "--manifest", manifest, "--mapping", mapping,
             "--input-tensor", vit["input_tensor"]])
        run([root / "build/soma-sim", "--hardware", root / config["architecture"]["base"],
             "--mapping", mapping, "--weights", npz, "--input", spikes, "--output", output])
        run([sys.executable, root / "tools/compare_vit_block00_reference.py",
             "--npz", npz, "--manifest", manifest, "--trace", output / "firing_trace.csv",
             "--json", output / "correctness.json", "--markdown", output / "correctness.md"])
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        correctness = json.loads((output / "correctness.json").read_text(encoding="utf-8"))
        records.append({
            "block": block, "nir": str(nir), "output": str(output),
            "completed": summary["completed"],
            "all_computational_layers_exact": correctness["all_computational_layers_exact"],
            "summary": summary,
        })

    additive_fields = ["hardware_latency_ps", "packets", "noc_hops", "synaptic_updates", "attention_updates"]
    energy_fields = ["axon", "router", "link", "memory", "synapse", "soma", "total"]
    cumulative = {field: sum(int(record["summary"][field]) for record in records) for field in additive_fields}
    cumulative["energy_pj"] = {field: sum(float(record["summary"]["energy_pj"][field]) for record in records)
                               for field in energy_fields}
    cumulative["host_latency_s"] = sum(float(record["summary"]["host_latency_s"]) for record in records)
    with h5py.File(nirs[-1], "r") as final_h5:
        final_terminal = terminal_node(final_h5)
    result = {
        "completed": all(record["completed"] for record in records),
        "all_computational_layers_exact": all(record["all_computational_layers_exact"] for record in records),
        "boundary_checks": boundaries,
        "blocks": records,
        "cumulative_sequential_work": cumulative,
        "final_model_output": {
            "block": blocks[-1], "nir_terminal_node": final_terminal,
            "simulator_output": str(output_root / f"block{blocks[-1]:02d}" / "correctness.json"),
        },
        "note": "累计值为六个 block 按串行样本边界分别运行的实际硬件工作量之和；每个 block 的 Cx State 仍按 NIR sample 边界 reset。",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "end_to_end_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"completed": result["completed"],
                      "all_computational_layers_exact": result["all_computational_layers_exact"],
                      "summary": str(output_root / "end_to_end_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()

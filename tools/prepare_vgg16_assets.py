#!/usr/bin/env python3
"""把远程已有的 VGG16 IF-SNN 参数转换为 SOMA-Sim 紧凑资产。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def source_plan(hin: int, win: int, hout: int, wout: int,
                kernel: Tuple[int, int], stride: Tuple[int, int],
                padding: Tuple[int, int]) -> Dict[str, np.ndarray]:
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    plan_pattern_id = np.empty(hin * win, dtype=np.int32)
    plan_dst_base = np.empty(hin * win, dtype=np.int32)
    pattern_ids: Dict[Tuple[Tuple[int, int], ...], int] = {}
    patterns: List[Tuple[Tuple[int, int], ...]] = []
    for iy in range(hin):
        for ix in range(win):
            contributions = []
            for ky in range(kh):
                numerator_y = iy + ph - ky
                if numerator_y % sh:
                    continue
                oy = numerator_y // sh
                if not 0 <= oy < hout:
                    continue
                for kx in range(kw):
                    numerator_x = ix + pw - kx
                    if numerator_x % sw:
                        continue
                    ox = numerator_x // sw
                    if 0 <= ox < wout:
                        contributions.append((oy * wout + ox, ky * kw + kx))
            contributions.sort()
            base = min((dst for dst, _ in contributions), default=0)
            signature = tuple((dst - base, kernel_index) for dst, kernel_index in contributions)
            pattern = pattern_ids.setdefault(signature, len(patterns))
            if pattern == len(patterns):
                patterns.append(signature)
            source = iy * win + ix
            plan_pattern_id[source] = pattern
            plan_dst_base[source] = base

    ptr = [0]
    dst_offset: List[int] = []
    kernel_index: List[int] = []
    for pattern in patterns:
        for dst, kernel in pattern:
            dst_offset.append(dst)
            kernel_index.append(kernel)
        ptr.append(len(dst_offset))
    return {
        "plan_pattern_id": plan_pattern_id,
        "plan_dst_base": plan_dst_base,
        "pattern_ptr": np.asarray(ptr, dtype=np.int32),
        "pattern_dst_offset": np.asarray(dst_offset, dtype=np.int32),
        "kernel_index": np.asarray(kernel_index, dtype=np.int64),
    }


def add_spatial(arrays: Dict[str, np.ndarray], prefix: str, weight: np.ndarray,
                hin: int, win: int, hout: int, wout: int,
                kernel: Tuple[int, int], stride: Tuple[int, int],
                padding: Tuple[int, int], cout: int, channelwise: bool = False) -> None:
    plan = source_plan(hin, win, hout, wout, kernel, stride, padding)
    arrays[f"{prefix}_weight"] = np.ascontiguousarray(weight, dtype=np.float32)
    for key in ("plan_pattern_id", "plan_dst_base", "pattern_ptr", "pattern_dst_offset"):
        arrays[f"{prefix}_{key}"] = plan[key]
    multiplier = 1 if channelwise else cout
    arrays[f"{prefix}_pattern_weight_offset"] = plan["kernel_index"] * multiplier


def layer_yaml(layer: Dict[str, object], router: int, next_id: str) -> List[str]:
    lines = [f"    - id: {layer['id']}", "      partition: aggregated", f"      op: {layer['op']}",
             f"      pe: {router}", "      core: 0", f"      router: {router}",
             f"      source_neurons: {layer['source_neurons']}", f"      neurons: {layer['neurons']}",
             f"      input_h: {layer.get('input_h', 1)}", f"      input_w: {layer.get('input_w', 1)}",
             f"      input_channels: {layer['input_channels']}",
             f"      output_h: {layer.get('output_h', 1)}", f"      output_w: {layer.get('output_w', 1)}",
             f"      output_channels: {layer['output_channels']}",
             f"      aggregate_core_count: {math.ceil(int(layer['neurons']) / 1024)}",
             f"      weight_prefix: {layer['id']}"]
    if layer["op"] != "linear" or layer["id"] != "readout":
        lines += ["      threshold: 1.0", "      leak: 1.0", "      reset: soft"]
    if layer.get("channelwise"):
        lines.append("      channelwise: true")
    if next_id:
        lines.append(f"      next: {next_id}")
    if layer["id"] == "readout":
        lines.append("      readout: true")
    return lines


def build_assets(source_path: Path, weights_path: Path, mapping_path: Path) -> None:
    source = np.load(source_path, allow_pickle=False)
    arrays: Dict[str, np.ndarray] = {}
    layers: List[Dict[str, object]] = []
    h = w = 32
    channels = 6
    conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    pools_after = {1, 3, 6, 9, 12}
    pool_index = 0
    for conv_index, output_channels in enumerate(conv_channels):
        prefix = f"conv{conv_index}"
        # Remote export 是 [Kh,Kw,Cin,Cout]；SOMA 热路径要求 [Cin,Kh,Kw,Cout]。
        ihwo = np.asarray(source[f"{prefix}_weight_hwio"]).transpose(2, 0, 1, 3)
        add_spatial(arrays, prefix, ihwo, h, w, h, w, (3, 3), (1, 1), (1, 1), output_channels)
        arrays[f"{prefix}_bias"] = np.asarray(source[f"{prefix}_bias"], dtype=np.float32)
        layers.append({"id": prefix, "op": "conv2d", "source_neurons": h*w*channels,
                       "neurons": h*w*output_channels, "input_h": h, "input_w": w,
                       "input_channels": channels, "output_h": h, "output_w": w,
                       "output_channels": output_channels})
        channels = output_channels
        if conv_index in pools_after:
            out_h, out_w = h // 2, w // 2
            pool = f"pool{pool_index}"
            add_spatial(arrays, pool, np.full(4, 0.25, dtype=np.float32), h, w, out_h, out_w,
                        (2, 2), (2, 2), (0, 0), channels, channelwise=True)
            layers.append({"id": pool, "op": "avgpool2d", "source_neurons": h*w*channels,
                           "neurons": out_h*out_w*channels, "input_h": h, "input_w": w,
                           "input_channels": channels, "output_h": out_h, "output_w": out_w,
                           "output_channels": channels, "channelwise": True})
            h, w = out_h, out_w
            pool_index += 1

    dense_specs = [("fc1", "fc1_weight_io", 512), ("fc2", "fc2_weight_io", 512),
                   ("readout", "readout_weight_io", 10)]
    source_neurons = h * w * channels
    for name, weight_key, output_neurons in dense_specs:
        arrays[f"{name}_weight"] = np.asarray(source[weight_key], dtype=np.float32)
        arrays[f"{name}_bias"] = np.asarray(source[f"{name}_bias"], dtype=np.float32)
        layers.append({"id": name, "op": "linear", "source_neurons": source_neurons,
                       "neurons": output_neurons, "input_channels": source_neurons,
                       "output_channels": output_neurons})
        source_neurons = output_neurons

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_path, **arrays)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["mapping:", "  model: vgg16_if_snn_cifar10", "  policy: aggregated_layer_partition_to_pe_core",
             "  hardware: arch/hardware.yaml", "  layers:", "    - id: input", "      partition: virtual",
             "      op: input", "      pe: 0", "      core: 0", "      router: 0", "      neurons: 6144",
             "      input_h: 32", "      input_w: 32", "      input_channels: 6", "      output_h: 32",
             "      output_w: 32", "      output_channels: 6", "      next: conv0", "      virtual_input: true"]
    for index, layer in enumerate(layers):
        next_id = str(layers[index + 1]["id"]) if index + 1 < len(layers) else ""
        lines.extend(layer_yaml(layer, index + 1, next_id))
    lines.append("  routes:")
    ids: Sequence[str] = ["input"] + [str(layer["id"]) for layer in layers]
    for index in range(len(ids) - 1):
        lines += [f"    - from: {ids[index]}", f"      to: {ids[index + 1]}",
                  f"      routers: [{index}, {index + 1}]"]
    mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {weights_path} ({weights_path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"wrote {mapping_path}")


def export_input(sample_path: Path, output_path: Path, timesteps: int, period_ps: int) -> None:
    sample = np.load(sample_path, allow_pickle=False)
    spikes = np.asarray(sample["input_spikes"][:timesteps], dtype=np.bool_)
    label = int(sample["label"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generated_time", "current_time", "spike_id", "timestep", "layer_id", "src_neuron",
              "src_pe", "src_router", "dst_pe", "dst_router", "route", "value", "expected_output"]
    spike_id = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        _, channels, _, width = spikes.shape
        for timestep in range(spikes.shape[0]):
            generated = timestep * period_ps
            for channel, y, x in np.argwhere(spikes[timestep]):
                neuron = (int(y) * width + int(x)) * channels + int(channel)
                writer.writerow({"generated_time": generated, "current_time": generated,
                                 "spike_id": spike_id, "timestep": timestep, "layer_id": "input",
                                 "src_neuron": neuron, "src_pe": 0, "src_router": 0,
                                 "dst_pe": 1, "dst_router": 1, "route": "debug_only",
                                 "value": 1.0, "expected_output": label})
                spike_id += 1
    print(f"wrote {spike_id} input spikes to {output_path}; label={label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parameters", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("input/vgg16_weights.npz"))
    parser.add_argument("--mapping", type=Path, default=Path("compiler/mapping_output/vgg16_mapping.yaml"))
    parser.add_argument("--input-csv", type=Path, default=Path("input/vgg16_input_spike.csv"))
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--timestep-period-ps", type=int, default=1_000_000_000_000)
    args = parser.parse_args()
    build_assets(args.source_parameters, args.weights, args.mapping)
    export_input(args.sample, args.input_csv, args.timesteps, args.timestep_period_ps)


if __name__ == "__main__":
    main()

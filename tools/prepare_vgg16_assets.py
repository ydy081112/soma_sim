#!/usr/bin/env python3
"""把远程已有的 VGG16 IF-SNN 参数转换为 SOMA-Sim 紧凑资产。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

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


def fuse_avgpool2x2_into_conv3x3(weight_hwio: np.ndarray) -> np.ndarray:
    """Compose AvgPool2d(2,2) -> Conv3x3 into Conv6x6(stride=2,padding=2)."""
    if weight_hwio.ndim != 4 or weight_hwio.shape[:2] != (3, 3):
        raise ValueError(f"expected HWIO 3x3 convolution, got {weight_hwio.shape}")
    # 每个原 kernel tap 覆盖 Pool 的四个输入位置；仍只保存规则 kernel，不展开连接。
    return np.repeat(np.repeat(weight_hwio, 2, axis=0), 2, axis=1) / 4.0


def fuse_avgpool2x2_into_dense_spatial_major(weight_io: np.ndarray) -> np.ndarray:
    """Compose the final 2x2 average pool into SOMA's spatial-major Dense rows."""
    if weight_io.ndim != 2:
        raise ValueError(f"expected Dense [Cin,Cout], got {weight_io.shape}")
    # logical source id 是 spatial * channels + channel；每个 spatial block 复用同一组 FC 行。
    return np.tile(weight_io, (4, 1)) / 4.0


CORES_PER_TILE = 4
MAX_NEURONS_PER_CORE = 1024
NOC_ROWS = 128


def layer_yaml(layer: Dict[str, object], start_core: int, next_id: str) -> List[str]:
    tile = start_core // CORES_PER_TILE
    core = start_core % CORES_PER_TILE
    core_count = math.ceil(int(layer["neurons"]) / MAX_NEURONS_PER_CORE)
    lines = [f"    - id: {layer['id']}", "      partition: contiguous_physical", f"      op: {layer['op']}",
             f"      pe: {tile}", f"      core: {core}", f"      router: {tile}",
             f"      source_neurons: {layer['source_neurons']}", f"      neurons: {layer['neurons']}",
             f"      input_h: {layer.get('input_h', 1)}", f"      input_w: {layer.get('input_w', 1)}",
             f"      input_channels: {layer['input_channels']}",
             f"      output_h: {layer.get('output_h', 1)}", f"      output_w: {layer.get('output_w', 1)}",
             f"      output_channels: {layer['output_channels']}",
             f"      aggregate_core_count: {core_count}",
             f"      physical_neuron_order: {'logical' if layer['op'] == 'linear' else 'channel_major'}",
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


def xy_route(source: int, destination: int) -> List[int]:
    x, y = divmod(source, NOC_ROWS)
    dx, dy = divmod(destination, NOC_ROWS)
    route = [source]
    while x != dx:
        x += 1 if x < dx else -1
        route.append(x * NOC_ROWS + y)
    while y != dy:
        y += 1 if y < dy else -1
        route.append(x * NOC_ROWS + y)
    return route


def build_assets(source_path: Path, weights_path: Path, mapping_path: Path) -> None:
    source = np.load(source_path, allow_pickle=False)
    arrays: Dict[str, np.ndarray] = {}
    layers: List[Dict[str, object]] = []
    h = w = 32
    channels = 6
    conv_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    pool_before_conv = {2, 4, 7, 10}
    for conv_index, output_channels in enumerate(conv_channels):
        prefix = f"conv{conv_index}"
        weight_hwio = np.asarray(source[f"{prefix}_weight_hwio"])
        output_h, output_w = h, w
        kernel = (3, 3)
        stride = (1, 1)
        padding = (1, 1)
        if conv_index in pool_before_conv:
            weight_hwio = fuse_avgpool2x2_into_conv3x3(weight_hwio)
            output_h, output_w = h // 2, w // 2
            kernel = (6, 6)
            stride = (2, 2)
            padding = (2, 2)

        # Remote export 是 [Kh,Kw,Cin,Cout]；SOMA 热路径要求 [Cin,Kh,Kw,Cout]。
        ihwo = weight_hwio.transpose(2, 0, 1, 3)
        add_spatial(arrays, prefix, ihwo, h, w, output_h, output_w,
                    kernel, stride, padding, output_channels)
        arrays[f"{prefix}_bias"] = np.asarray(source[f"{prefix}_bias"], dtype=np.float32)
        layers.append({"id": prefix, "op": "conv2d", "source_neurons": h*w*channels,
                       "neurons": output_h*output_w*output_channels,
                       "input_h": h, "input_w": w,
                       "input_channels": channels, "output_h": output_h,
                       "output_w": output_w,
                       "output_channels": output_channels})
        h, w = output_h, output_w
        channels = output_channels

    dense_specs = [("fc1", "fc1_weight_io", 512), ("fc2", "fc2_weight_io", 512),
                   ("readout", "readout_weight_io", 10)]
    source_neurons = h * w * channels
    for name, weight_key, output_neurons in dense_specs:
        dense_weight = np.asarray(source[weight_key], dtype=np.float32)
        if name == "fc1":
            if (h, w) != (2, 2) or dense_weight.shape[0] != channels:
                raise ValueError("final Pool->FC fusion expects a 2x2 spatial input")
            dense_weight = fuse_avgpool2x2_into_dense_spatial_major(dense_weight)
        arrays[f"{name}_weight"] = np.ascontiguousarray(dense_weight, dtype=np.float32)
        arrays[f"{name}_bias"] = np.asarray(source[f"{name}_bias"], dtype=np.float32)
        layers.append({"id": name, "op": "linear", "source_neurons": source_neurons,
                       "neurons": output_neurons, "input_channels": source_neurons,
                       "output_channels": output_neurons})
        source_neurons = output_neurons

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_path, **arrays)

    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["mapping:", "  model: vgg16_if_snn_cifar10", "  policy: contiguous_physical_core_partitions",
             "  hardware: arch/hardware.yaml", "  layers:", "    - id: input", "      partition: virtual",
             "      op: input", "      pe: 0", "      core: 0", "      router: 0", "      neurons: 6144",
             "      input_h: 32", "      input_w: 32", "      input_channels: 6", "      output_h: 32",
             "      output_w: 32", "      output_channels: 6", "      aggregate_core_count: 6",
             "      physical_neuron_order: channel_major", "      next: conv0", "      virtual_input: true"]
    core_cursor = 6
    placements = [("input", 0)]
    for index, layer in enumerate(layers):
        next_id = str(layers[index + 1]["id"]) if index + 1 < len(layers) else ""
        lines.extend(layer_yaml(layer, core_cursor, next_id))
        placements.append((str(layer["id"]), core_cursor // CORES_PER_TILE))
        core_cursor += math.ceil(int(layer["neurons"]) / MAX_NEURONS_PER_CORE)
    lines.append("  routes:")
    for index in range(len(placements) - 1):
        source_id, source_router = placements[index]
        target_id, target_router = placements[index + 1]
        route = ", ".join(str(router) for router in xy_route(source_router, target_router))
        lines += [f"    - from: {source_id}", f"      to: {target_id}",
                  f"      routers: [{route}]"]
    mapping_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {weights_path} ({weights_path.stat().st_size / 1024**2:.1f} MiB)")
    print(f"wrote {mapping_path}")


def export_input(sample_path: Path, output_path: Path, timesteps: int) -> None:
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
        for timestep_index in range(spikes.shape[0]):
            timestep = timestep_index + 1
            for channel, y, x in np.argwhere(spikes[timestep_index]):
                neuron = (int(y) * width + int(x)) * channels + int(channel)
                writer.writerow({"generated_time": 0, "current_time": 0,
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
    args = parser.parse_args()
    build_assets(args.source_parameters, args.weights, args.mapping)
    export_input(args.sample, args.input_csv, args.timesteps)


if __name__ == "__main__":
    main()

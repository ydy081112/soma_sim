#!/usr/bin/env python3
"""生成与 SANA-FE 已完成的 ResNet18 T=16 block 一一对应的紧凑资产。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prepare_vgg16_assets import add_spatial, xy_route


TIMESTEPS = 16
CORES_PER_TILE = 4
MAX_NEURONS_PER_CORE = 1024


@dataclass(frozen=True)
class Block:
    stem: str
    block: str
    parameter_prefix: str | None
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, ...]
    kind: str
    downsample: bool = False


BLOCKS = (
    Block("B0_stem", "B0", None, (6, 224, 224), (64, 112, 112), "stem"),
    Block("B1_2_layer1_0_conv2_residual", "B1.2", "layer1_0",
          (64, 56, 56), (64, 56, 56), "split"),
    Block("B2_layer1_1", "B2", "layer1_1", (64, 56, 56), (64, 56, 56), "basic"),
    Block("B3_layer2_0", "B3", "layer2_0", (64, 56, 56), (128, 28, 28), "basic", True),
    Block("B4_layer2_1", "B4", "layer2_1", (128, 28, 28), (128, 28, 28), "basic"),
    Block("B5_layer3_0", "B5", "layer3_0", (128, 28, 28), (256, 14, 14), "basic", True),
    Block("B6_layer3_1", "B6", "layer3_1", (256, 14, 14), (256, 14, 14), "basic"),
    Block("B7_layer4_0", "B7", "layer4_0", (256, 14, 14), (512, 7, 7), "basic", True),
    Block("B8_layer4_1", "B8", "layer4_1", (512, 7, 7), (512, 7, 7), "basic"),
    Block("B9_head", "B9", None, (512, 7, 7), (1000,), "head"),
)


def f32(value) -> np.ndarray:
    """复现 SANA-FE Python binding 的 float32 参数传输。"""
    return np.asarray(value, dtype=np.float32)


def core_count(neurons: int) -> int:
    return math.ceil(neurons / MAX_NEURONS_PER_CORE)


def layer_lines(layer_id: str, op: str, shape: tuple[int, ...], source_neurons: int,
                first_core: int, weight_prefix: str, *, virtual: bool = False,
                readout: bool = False) -> list[str]:
    if len(shape) == 3:
        channels, height, width = shape
    else:
        channels, height, width = shape[0], 1, 1
    neurons = math.prod(shape)
    lines = [
        f"    - id: {layer_id}",
        f"      partition: {'virtual' if virtual else 'contiguous_physical'}",
        f"      op: {op}",
        f"      pe: {first_core // CORES_PER_TILE}",
        f"      core: {first_core % CORES_PER_TILE}",
        f"      router: {first_core // CORES_PER_TILE}",
        f"      neurons: {neurons}",
        f"      input_h: {height}",
        f"      input_w: {width}",
        f"      input_channels: {channels}",
        f"      output_h: {height}",
        f"      output_w: {width}",
        f"      output_channels: {channels}",
        f"      aggregate_core_count: {core_count(neurons)}",
        "      physical_neuron_order: channel_major",
        f"      weight_prefix: {weight_prefix}",
    ]
    if virtual:
        lines.append("      virtual_input: true")
    else:
        lines.append(f"      source_neurons: {source_neurons}")
        lines += [
            "      threshold: 1.0",
            "      leak: 1.0",
            "      reset: soft",
            "      membrane_quantization_step: 0.015625",
            "      threshold_comparison: greater",
        ]
    if readout:
        lines.append("      readout: true")
    return lines


def connection_lines(source: str, target: str, kind: str, prefix: str, delay: int,
                     *, hardware_type: str | None = None,
                     channelwise: bool = False) -> list[str]:
    lines = [f"    - from: {source}", f"      to: {target}", f"      type: {kind}"]
    if hardware_type is not None:
        lines.append(f"      hardware_type: {hardware_type}")
    lines += [f"      weight_prefix: {prefix}", f"      delay: {delay}"]
    if channelwise:
        lines.append("      channelwise: true")
    return lines


def route_lines(source: str, target: str, source_core: int, target_core: int) -> list[str]:
    routers = ", ".join(map(str, xy_route(source_core // 4, target_core // 4)))
    return [f"    - from: {source}", f"      to: {target}", f"      routers: [{routers}]"]


def write_mapping(path: Path, spec: Block, layers: list[dict], connections: list[dict]) -> None:
    lines = ["mapping:", f"  model: resnet18_imagenet_{spec.stem}_t16",
             "  policy: sanafe_sequential_contiguous_cores",
             "  hardware: arch/resnet18_sanafe.yaml",
             "  layers:"]
    for layer in layers:
        lines += layer_lines(**layer)
    lines.append("  connections:")
    for connection in connections:
        lines += connection_lines(**connection)
    lines.append("  routes:")
    first_core = {layer["layer_id"]: layer["first_core"] for layer in layers}
    for connection in connections:
        lines += route_lines(connection["source"], connection["target"],
                             first_core[connection["source"]], first_core[connection["target"]])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def spatial(arrays: dict[str, np.ndarray], prefix: str, weight_hwio: np.ndarray,
            source_shape: tuple[int, int, int], output_shape: tuple[int, int, int],
            kernel: int, stride: int, padding: int, *, channelwise: bool = False) -> None:
    cin, hin, win = source_shape
    cout, hout, wout = output_shape
    weight = f32(weight_hwio)
    if not channelwise:
        weight = weight.transpose(2, 0, 1, 3)
    add_spatial(arrays, prefix, weight, hin, win, hout, wout,
                (kernel, kernel), (stride, stride), (padding, padding), cout, channelwise)


def build_block(spec: Block, parameters, boundary, weights_path: Path, mapping_path: Path) -> dict:
    arrays: dict[str, np.ndarray] = {}
    layers: list[dict] = []
    connections: list[dict] = []
    cursor = 0

    def add_layer(layer_id: str, op: str, shape: tuple[int, ...], source_neurons: int,
                  weight_prefix: str, *, virtual: bool = False, readout: bool = False) -> int:
        nonlocal cursor
        first = cursor
        layers.append(dict(layer_id=layer_id, op=op, shape=shape,
                           source_neurons=source_neurons, first_core=first,
                           weight_prefix=weight_prefix, virtual=virtual, readout=readout))
        cursor += core_count(math.prod(shape))
        return first

    if spec.kind == "stem":
        add_layer("input", "input", spec.input_shape, 0, "input", virtual=True)
        add_layer("output", "conv2d", spec.output_shape, math.prod(spec.input_shape), "output")
        spatial(arrays, "stem", parameters["stem_weight_hwio"], spec.input_shape,
                spec.output_shape, 7, 2, 3)
        arrays["output_bias"] = f32(parameters["stem_bias"])
        connections.append(dict(source="input", target="output", kind="spatial",
                                prefix="stem", delay=0))
    elif spec.kind == "split":
        skip_shape = (64, 112, 112)
        add_layer("main_input", "input", spec.input_shape, 0, "main_input", virtual=True)
        add_layer("skip_input", "input", skip_shape, 0, "skip_input", virtual=True)
        add_layer("output", "conv2d", spec.output_shape, math.prod(spec.input_shape), "output")
        spatial(arrays, "main", parameters["layer1_0_conv2_weight_hwio"],
                spec.input_shape, spec.output_shape, 3, 1, 1)
        identity_scale = float(f32(parameters["layer1_0_identity_scale"]))
        skip_weight = np.full((3, 3), np.float32(identity_scale / 9.0),
                              dtype=np.float32)
        spatial(arrays, "skip", skip_weight, skip_shape, spec.output_shape, 3, 2, 1,
                channelwise=True)
        arrays["output_bias"] = f32(parameters["layer1_0_conv2_bias"])
        connections += [
            dict(source="main_input", target="output", kind="spatial", prefix="main", delay=0),
            dict(source="skip_input", target="output", kind="spatial", prefix="skip", delay=1,
                 hardware_type="identity", channelwise=True),
        ]
    elif spec.kind == "basic":
        prefix = spec.parameter_prefix
        assert prefix is not None
        add_layer("input", "input", spec.input_shape, 0, "input", virtual=True)
        add_layer("conv1", "conv2d", spec.output_shape, math.prod(spec.input_shape), "conv1")
        add_layer("output", "conv2d", spec.output_shape, math.prod(spec.output_shape), "output")
        stride = 2 if spec.downsample else 1
        spatial(arrays, "conv1_conn", parameters[f"{prefix}_conv1_weight_hwio"],
                spec.input_shape, spec.output_shape, 3, stride, 1)
        spatial(arrays, "conv2_conn", parameters[f"{prefix}_conv2_weight_hwio"],
                spec.output_shape, spec.output_shape, 3, 1, 1)
        arrays["conv1_bias"] = f32(parameters[f"{prefix}_conv1_bias"])
        post_bias = f32(parameters[f"{prefix}_conv2_bias"])
        connections += [
            dict(source="input", target="conv1", kind="spatial", prefix="conv1_conn", delay=0),
            dict(source="conv1", target="output", kind="spatial", prefix="conv2_conn", delay=0),
        ]
        if spec.downsample:
            spatial(arrays, "skip", parameters[f"{prefix}_downsample_weight_hwio"],
                    spec.input_shape, spec.output_shape, 1, 2, 0)
            post_bias = f32(post_bias.astype(np.float64) +
                            f32(parameters[f"{prefix}_downsample_bias"]).astype(np.float64))
            connections.append(dict(source="input", target="output", kind="spatial",
                                    prefix="skip", delay=1))
        else:
            arrays["skip_weight"] = f32(parameters[f"{prefix}_identity_scale"]).reshape(1)
            connections.append(dict(source="input", target="output", kind="identity",
                                    prefix="skip", delay=1, hardware_type="identity"))
        arrays["output_bias"] = post_bias
    else:
        add_layer("input", "input", spec.input_shape, 0, "input", virtual=True)
        add_layer("readout", "linear", spec.output_shape, math.prod(spec.input_shape),
                  "readout", readout=True)
        channels, height, width = spec.input_shape
        arrays["readout_weight"] = f32(parameters["readout_weight_io"]).reshape(
            channels, height, width, spec.output_shape[0]).transpose(1, 2, 0, 3).reshape(
                math.prod(spec.input_shape), spec.output_shape[0])
        arrays["readout_bias"] = f32(parameters["readout_bias"])
        connections.append(dict(source="input", target="readout", kind="dense",
                                prefix="readout", delay=0))

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(weights_path, **arrays)
    write_mapping(mapping_path, spec, layers, connections)
    output_key = None if spec.kind == "head" else "output_spikes"
    return {
        "block": spec.block,
        "stem": spec.stem,
        "kind": spec.kind,
        "timesteps": TIMESTEPS,
        "mapped_core_count": cursor,
        "output_layer": "readout" if spec.kind == "head" else "output",
        "reference_output_spikes": (None if output_key is None else
                                    int(np.asarray(boundary[output_key][:TIMESTEPS]).sum())),
        # boundary scalar 是完整 128-step 预测；block T=16 应以第 16 个 potential 为准。
        "reference_prediction": (int(np.asarray(boundary["reference_potential"])[TIMESTEPS - 1].argmax())
                                 if spec.kind == "head" else None),
    }


def write_input(path: Path, sources: list[tuple[str, np.ndarray]], expected: int | None) -> int:
    fields = ["generated_time", "current_time", "spike_id", "timestep", "layer_id",
              "src_neuron", "src_pe", "src_router", "dst_pe", "dst_router", "route",
              "value", "expected_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    spike_id = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timestep in range(TIMESTEPS):
            for layer_id, trace in sources:
                _, channels, _, width = trace.shape
                for channel, y, x in np.argwhere(trace[timestep]):
                    logical = (int(y) * width + int(x)) * channels + int(channel)
                    writer.writerow({"generated_time": 0, "current_time": 0, "spike_id": spike_id,
                                     "timestep": timestep + 1, "layer_id": layer_id,
                                     "src_neuron": logical, "src_pe": 0, "src_router": 0,
                                     "dst_pe": 0, "dst_router": 0, "route": "debug_only",
                                     "value": 1.0, "expected_output": "" if expected is None else expected})
                    spike_id += 1
    return spike_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("input/resnet18_source"))
    parser.add_argument("--asset-dir", type=Path, default=Path("input/resnet18_blocks"))
    parser.add_argument("--mapping-dir", type=Path,
                        default=Path("compiler/mapping_output/resnet18_blocks"))
    parser.add_argument("--blocks", nargs="*", default=[block.block for block in BLOCKS])
    args = parser.parse_args()
    selected = set(args.blocks)
    unknown = selected - {block.block for block in BLOCKS}
    if unknown:
        raise SystemExit(f"unknown blocks: {sorted(unknown)}")
    parameters = np.load(args.source_dir / "snn_parameters.npz", allow_pickle=False)
    manifest = []
    for spec in BLOCKS:
        if spec.block not in selected:
            continue
        boundary = np.load(args.source_dir / "boundaries" / f"{spec.stem}.npz", allow_pickle=False)
        block_dir = args.asset_dir / spec.stem
        report = build_block(spec, parameters, boundary, block_dir / "weights.npz",
                             args.mapping_dir / f"{spec.stem}.yaml")
        if spec.kind == "split":
            sources = [("main_input", np.asarray(boundary["main_input_spikes"])),
                       ("skip_input", np.asarray(boundary["skip_input_spikes"]))]
        else:
            sources = [("input", np.asarray(boundary["input_spikes"]))]
        expected = report["reference_prediction"]
        report["input_spikes"] = write_input(block_dir / "input_spike.csv", sources, expected)
        manifest.append(report)
        print(f"{spec.block}: cores={report['mapped_core_count']} input_spikes={report['input_spikes']}")
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    (args.asset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                   encoding="utf-8")


if __name__ == "__main__":
    main()

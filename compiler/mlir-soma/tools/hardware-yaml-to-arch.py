#!/usr/bin/env python3
"""将 SOMA hardware.yaml 的 core-level 字段导出为 noc / snn_arch MLIR。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

DURATION = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:ps|ns|us|ms|s)$")


def require_map(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label}: expected mapping")
    return value


def require_string(mapping, key, label):
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key}: expected non-empty string")
    return value


def require_positive_int(mapping, key, label):
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label}.{key}: expected positive integer")
    return value


def parse_duration(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label}: expected duration string with unit")
    duration = value.replace(" ", "")
    if not DURATION.fullmatch(duration):
        raise ValueError(f"{label}: expected non-negative ps/ns/us/ms/s duration")
    return duration


def convert(source: Path, destination: Path, noc_symbol: str, core_symbol: str):
    with source.open(encoding="utf-8") as stream:
        document = require_map(yaml.safe_load(stream), str(source))
    architecture = require_map(document.get("architecture"), "architecture")
    noc = require_map(architecture.get("noc"), "architecture.noc")
    core = require_map(architecture.get("core"), "architecture.core")
    link = require_map(noc.get("link"), "architecture.noc.link")

    topology = require_string(noc, "topology", "architecture.noc")
    routing = require_string(noc, "routing", "architecture.noc")
    if topology != "mesh":
        raise ValueError(f"architecture.noc.topology: core-level frontend supports mesh, got {topology!r}")
    if routing != "xy":
        raise ValueError(f"architecture.noc.routing: core-level frontend supports xy, got {routing!r}")
    rows = require_positive_int(noc, "rows", "architecture.noc")
    columns = require_positive_int(noc, "cols", "architecture.noc")
    hop_latency = parse_duration(link.get("hardware_latency"),
                                 "architecture.noc.link.hardware_latency")
    neuron_capacity = require_positive_int(core, "max_neurons", "architecture.core")

    text = "\n".join([
        "module {",
        f"  noc.network @{noc_symbol} {{topology = {json.dumps(topology)}, dimensions = [{rows}, {columns}], hop_latency = {hop_latency}, routing = {json.dumps(routing)}}}",
        f"  snn_arch.core_type @{core_symbol} {{neuron_capacity = {neuron_capacity}}}",
        "}",
        "",
    ])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noc-symbol", default="noc0")
    parser.add_argument("--core-symbol", default="standard_core")
    args = parser.parse_args()
    convert(args.input, args.output, args.noc_symbol, args.core_symbol)


if __name__ == "__main__":
    main()

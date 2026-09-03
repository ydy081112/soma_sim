#!/usr/bin/env python3
"""把 NeMo nfg1/SQLite 转成 SOMA 的紧凑 TrueNorth crossbar 资产。"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORES = 19_860
AXONS = 256
NEURONS = 256
SLOTS = CORES * NEURONS
EXPECTED_ACTIVE = 2_944_798
EXPECTED_CONNECTIONS = 241_313_977
EXPECTED_ROUTES = 2_944_788

G_RE = re.compile(r"G\[(\d+)\]=\{(.*)\}$")
K_RE = re.compile(r'K\[(\d+)\]="([0-9 ]*)"$')
R_RE = re.compile(r"R\[(\d+)\]=")


def memmap(path: Path, dtype: str, shape: tuple[int, ...], fill: int | None = None) -> np.memmap:
    array = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill is not None:
        array.fill(fill)
    return array


def flush_core(rows: np.memmap, core: int, patterns: np.ndarray) -> None:
    if core < 0:
        return
    # nfg1 按 destination neuron 保存 256-bit connectivity；运行时转置为 axon-major。
    bits = np.unpackbits(patterns.view(np.uint8).reshape(NEURONS, 32), axis=1,
                         bitorder="little")
    packed = np.packbits(bits.T, axis=1, bitorder="little")
    rows[core] = np.ascontiguousarray(packed).view("<u8").reshape(AXONS, 4)


def convert_model(model: Path, work: Path) -> tuple[dict[str, np.memmap], dict[str, int]]:
    active = memmap(work / "active.bin", "u1", (SLOTS,), 0)
    # NeMo 对未映射 slot 不做初始化；其 threshold 保持 calloc 得到的 0。
    threshold = memmap(work / "threshold.bin", "<f4", (SLOTS,), 0)
    bias = memmap(work / "bias.bin", "<f4", (SLOTS,), 0)
    neuron_weight = memmap(work / "weight.bin", "<i2", (SLOTS, 4), 0)
    axon_type = memmap(work / "axon_type.bin", "u1", (CORES, AXONS), 0)
    rows = memmap(work / "rows.bin", "<u8", (CORES, AXONS, 4), 0)
    route_partition = memmap(work / "route_partition.bin", "<i4", (SLOTS,), -1)
    route_axon = memmap(work / "route_axon.bin", "<u2", (SLOTS,), 0)

    patterns: dict[int, int] = {}
    current_core = -1
    core_patterns = np.zeros((NEURONS, 4), dtype="<u8")
    active_count = connection_count = route_count = 0
    with model.open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.rstrip("\n")
            match = G_RE.match(line)
            if match:
                core = int(match.group(1))
                values = np.fromstring(match.group(2), dtype=np.uint8, sep=",")
                if core >= CORES or values.size != AXONS or np.any(values > 3):
                    raise ValueError(f"invalid G row for core {core}")
                axon_type[core] = values
                continue
            match = K_RE.match(line)
            if match:
                value = 0
                text = match.group(2)
                for axon in map(int, text.split()) if text else ():
                    value |= 1 << axon
                patterns[int(match.group(1))] = value
                continue
            match = R_RE.match(line)
            if match:
                flush_core(rows, current_core, core_patterns)
                current_core = int(match.group(1))
                core_patterns.fill(0)
                continue
            if current_core < 0 or not line or line == "]=]":
                continue
            fields = line.split("|")
            if len(fields) != 10:
                continue
            local, pattern_id = int(fields[0]), int(fields[1])
            weights = tuple(map(int, fields[2].split(",")))
            alpha, leak_sign, leak = map(int, fields[3:6])
            destination_core, destination_axon = map(int, fields[6:8])
            output, gamma = map(int, fields[8:10])
            if len(weights) != 4 or gamma != 1 or output not in (0, 1):
                raise ValueError(f"unsupported neuron record: {line}")
            physical = current_core * NEURONS + local
            active[physical] = 1
            threshold[physical] = alpha
            bias[physical] = leak_sign * leak
            neuron_weight[physical] = weights
            if destination_core >= 0:
                route_partition[physical] = destination_core
                route_axon[physical] = destination_axon
                route_count += 1
            pattern = patterns[pattern_id]
            core_patterns[local] = tuple((pattern >> (64 * word)) & ((1 << 64) - 1)
                                         for word in range(4))
            active_count += 1
            connection_count += pattern.bit_count()
    flush_core(rows, current_core, core_patterns)
    counts = {"cores": CORES, "active_neurons": active_count,
              "enabled_connections": connection_count, "routes": route_count}
    expected = {"active_neurons": EXPECTED_ACTIVE,
                "enabled_connections": EXPECTED_CONNECTIONS, "routes": EXPECTED_ROUTES}
    for key, value in expected.items():
        if counts[key] != value:
            raise RuntimeError(f"{key}: expected {value}, got {counts[key]}")
    return ({"tn_active": active, "tn_threshold": threshold, "tn_bias": bias,
             "tn_neuron_weight": neuron_weight, "tn_axon_type": axon_type,
             "tn_crossbar_rows": rows,
             "tn_recurrent_route_destination_partition": route_partition,
             "tn_recurrent_route_destination_axon": route_axon}, counts)


def convert_input(database: Path, csv_path: Path, work: Path,
                  max_timestep: int) -> tuple[dict[str, np.memmap], int]:
    with sqlite3.connect(database) as connection:
        count = connection.execute(
            "SELECT count(*) FROM input_spikes WHERE time <= ?", (max_timestep,)).fetchone()[0]
        route_partition = memmap(work / "input_partition.bin", "<i4", (count,), -1)
        route_axon = memmap(work / "input_axon.bin", "<u2", (count,), 0)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="ascii") as output:
            writer = csv.writer(output)
            writer.writerow(["generated_time", "current_time", "spike_id", "timestep",
                             "layer_id", "src_neuron", "value", "expected_output"])
            cursor = connection.execute(
                "SELECT time, core, axon FROM input_spikes WHERE time <= ? ORDER BY rowid",
                (max_timestep,))
            for index, (timestep, core, axon) in enumerate(cursor):
                route_partition[index] = core
                route_axon[index] = axon
                writer.writerow([0, 0, index, timestep, "input", index, 1, 3])
    return ({"tn_input_route_destination_partition": route_partition,
             "tn_input_route_destination_axon": route_axon}, count)


def update_mapping(path: Path, input_neurons: int) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(    - id: input\n(?:.*\n)*?      neurons:) \d+",
                  rf"\g<1> {input_neurons}", text, count=1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path,
                        default=ROOT / "input/truenorth_vgg16_nemo_model.nfg1")
    parser.add_argument("--spikes", type=Path,
                        default=ROOT / "input/truenorth_vgg16_nemo_spikes.sqlite")
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "input/truenorth_vgg16_weights.npz")
    parser.add_argument("--input", type=Path,
                        default=ROOT / "input/truenorth_vgg16_input_spike_t16.csv")
    parser.add_argument("--mapping", type=Path,
                        default=ROOT / "compiler/mapping_output/truenorth_vgg16_mapping.yaml")
    parser.add_argument("--timesteps", type=int, default=16)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="soma_tn_", dir=args.weights.parent) as temp:
        work = Path(temp)
        arrays, counts = convert_model(args.model, work)
        input_arrays, input_count = convert_input(args.spikes, args.input, work, args.timesteps)
        arrays.update(input_arrays)
        np.savez(args.weights, **arrays)
    update_mapping(args.mapping, input_count)
    print({**counts, "input_rows": input_count, "timesteps": args.timesteps})


if __name__ == "__main__":
    main()

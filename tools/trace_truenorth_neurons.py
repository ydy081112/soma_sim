#!/usr/bin/env python3
"""离线重放少量 TrueNorth neuron，定位 NeMo/SOMA 状态差异。"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np


def parse_targets(text: str) -> list[tuple[int, int]]:
    targets: list[tuple[int, int]] = []
    for item in text.split(","):
        core, local = item.split(":", maxsplit=1)
        targets.append((int(core), int(local)))
    return targets


def active_fire_records(path: Path, targets: set[tuple[int, int]]) -> dict[tuple[int, int], str]:
    records: dict[tuple[int, int], str] = {}
    with path.open(encoding="ascii") as stream:
        next(stream)
        for line in stream:
            fields = line.rstrip().split(",")
            if len(fields) < 3:
                continue
            timestamp, core, local = fields[:3]
            key = (int(core), int(local))
            if 4.0 <= float(timestamp) < 5.0 and key in targets:
                records.setdefault(key, timestamp)
    return records


def trace_target(weights: np.lib.npyio.NpzFile, database: sqlite3.Connection,
                 core: int, local: int, input_max_timestep: int,
                 threshold_compare_mode: str) -> list[str]:
    slot = core * 256 + local
    threshold = int(weights["tn_threshold"][slot])
    leak_per_tick = int(weights["tn_bias"][slot])
    neuron_weights = weights["tn_neuron_weight"][slot]
    rows = weights["tn_crossbar_rows"]
    axon_types = weights["tn_axon_type"]
    events = database.execute(
        "SELECT rowid, time, axon FROM input_spikes "
        "WHERE core = ? AND time <= ? ORDER BY time, rowid",
        (core, input_max_timestep),
    ).fetchall()

    contributions: list[tuple[int, int, int, int, int, int]] = []
    for rowid, time, axon in events:
        connected = (int(rows[core, axon, local // 64]) >> (local % 64)) & 1
        axon_type = int(axon_types[core, axon])
        configured_weight = int(neuron_weights[axon_type])
        # NeMo TNIntegrate 将任意非零有效权重转成 +1。
        contribution = int(connected and configured_weight != 0)
        contributions.append((rowid, time, axon, axon_type, configured_weight, contribution))

    synaptic_sum = sum(item[-1] for item in contributions)
    # t2 input 的 heartbeat 在 big tick 3 执行；lastLeakTime 初始为 big tick 0。
    elapsed_ticks = input_max_timestep + 1
    membrane_start = 0
    soma_leak_before = membrane_start
    soma_leak_after = soma_leak_before  # 当前 mapping leak=1，不缩放历史膜电位。
    membrane_after_input = soma_leak_after + synaptic_sum
    membrane_after_leak = membrane_after_input + elapsed_ticks * leak_per_tick
    if threshold_compare_mode == "unsigned_promotion":
        # 对应 SOMA 的可配置 NeMo 兼容比较：int32 左值提升成 uint32 再比较。
        promoted_membrane = membrane_after_leak & 0xFFFF_FFFF
        fired = promoted_membrane >= threshold
        comparison = f"uint32({membrane_after_leak})={promoted_membrane} >= {threshold}"
    else:
        fired = membrane_after_leak >= threshold
        comparison = f"{membrane_after_leak} >= {threshold}"
    membrane_after_reset = membrane_after_leak - threshold if fired else membrane_after_leak

    lines = [
        f"## Core {core}, local neuron {local}",
        "",
        f"- active slot: `{int(weights['tn_active'][slot])}`",
        f"- SOMA logical processing: input through t{input_max_timestep} -> heartbeat/process t{input_max_timestep + 1} -> report t{input_max_timestep + 2}",
        f"- threshold: `{threshold}`; NeMo leak sign×magnitude: `{leak_per_tick}` per big tick",
        f"- timestep start membrane: `{membrane_start}`",
        "",
        "| SQLite row | input timestep | axon | axon type | configured weight | NeMo/SOMA contribution |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {rowid} | {time} | {axon} | {axon_type} | {weight} | {contribution} |"
        for rowid, time, axon, axon_type, weight, contribution in contributions
    )
    lines.extend([
        "",
        f"- synaptic input accumulated after `TNIntegrate` / SOMA crossbar: `{synaptic_sum}`",
        f"- membrane before/after SOMA multiplicative leak: `{soma_leak_before}` -> `{soma_leak_after}`",
        f"- membrane after synaptic input: `{membrane_after_input}`",
        f"- NeMo `TNNumericLeakCalc` tick count: `{elapsed_ticks}`; membrane after integer leak: `{membrane_after_leak}`",
        f"- threshold check `{comparison}`: `{'fire' if fired else 'no fire'}`",
        f"- reset result: `{membrane_after_reset}`; final committed state expected from source sequence: `{membrane_after_reset}`",
        "",
    ])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path,
                        default=Path("input/truenorth_vgg16_weights.npz"))
    parser.add_argument("--spikes", type=Path,
                        default=Path("input/truenorth_vgg16_nemo_spikes.sqlite"))
    parser.add_argument("--targets", default="919:63,988:63,521:63")
    parser.add_argument("--input-max-timestep", type=int, default=2)
    parser.add_argument("--threshold-compare-mode",
                        choices=("signed", "unsigned_promotion"), default="signed")
    parser.add_argument("--nemo-fire-record", type=Path)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    records = (active_fire_records(args.nemo_fire_record, set(targets))
               if args.nemo_fire_record else {})
    weights = np.load(args.weights)
    with sqlite3.connect(args.spikes) as database:
        print("# TrueNorth t4 active-neuron state trace")
        print()
        print("NeMo source sequence is `SYNAPSE_OUT: TNIntegrate`, then one next-big-tick "
              "heartbeat: `TNNumericLeakCalc -> ringing -> TNfireFloorCelingReset -> commit`.")
        print()
        for core, local in targets:
            if (core, local) in records:
                print(f"NeMo fire record: `{records[(core, local)]}` (reported t4).")
            else:
                print("NeMo fire record: target not found in [4, 5).")
            print("\n".join(trace_target(weights, database, core, local,
                                        args.input_max_timestep,
                                        args.threshold_compare_mode)))


if __name__ == "__main__":
    main()

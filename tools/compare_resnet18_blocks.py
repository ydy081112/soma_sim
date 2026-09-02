#!/usr/bin/env python3
"""汇总相同 T=16 block 的 SOMA-Sim / SANA-FE latency、energy 和 host 数据。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def relative_percent(actual: float, reference: float) -> float | None:
    return None if reference == 0.0 else (actual / reference - 1.0) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, default=Path("input/resnet18_blocks"))
    parser.add_argument("--sana-dir", type=Path,
                        default=Path("input/resnet18_source/sanafe_results"))
    parser.add_argument("--boundary-dir", type=Path,
                        default=Path("input/resnet18_source/boundaries"))
    parser.add_argument("--soma-dir", type=Path, default=Path("output/resnet18_blocks_t16"))
    parser.add_argument("--output", type=Path,
                        default=Path("output/resnet18_blocks_t16/comparison.csv"))
    args = parser.parse_args()

    manifests = json.loads((args.asset_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for item in manifests:
        stem = item["stem"]
        soma_path = args.soma_dir / stem / "summary.json"
        if not soma_path.exists():
            continue
        soma = json.loads(soma_path.read_text(encoding="utf-8"))
        with (args.sana_dir / f"{stem}.csv").open(newline="", encoding="utf-8") as handle:
            sana_rows = list(csv.DictReader(handle))
        if len(sana_rows) != 16:
            raise RuntimeError(f"{stem}: expected 16 SANA-FE rows, got {len(sana_rows)}")
        sana_latency = sum(float(row["latency_s"]) for row in sana_rows)
        sana_energy = sum(float(row["energy_total_j"]) for row in sana_rows)
        sana_host = sum(float(row["host_runtime_s"]) for row in sana_rows)
        soma_energy = float(soma["energy_pj"]["total"]) * 1.0e-12
        with (args.soma_dir / stem / "layer_metrics.csv").open(newline="", encoding="utf-8") as handle:
            layer_rows = {row["layer_id"]: row for row in csv.DictReader(handle)}
        actual_output_spikes = int(layer_rows[item["output_layer"]]["emitted_spikes"])
        reference_output_spikes = item["reference_output_spikes"]
        functional_metric = "output_spike_count"
        functional_max_abs = None
        functional_match = (actual_output_spikes == reference_output_spikes)
        if item["kind"] == "head":
            boundary = np.load(args.boundary_dir / f"{stem}.npz", allow_pickle=False)
            reference = np.asarray(boundary["reference_potential"][15], dtype=np.float64)
            actual = np.asarray(soma["output_scores"], dtype=np.float64)
            functional_metric = "readout_potential"
            functional_max_abs = float(np.max(np.abs(actual - reference)))
            functional_match = bool(np.array_equal(actual.astype(np.float32),
                                                    reference.astype(np.float32)))
            reference_output_spikes = None
        rows.append({
            "block": item["block"],
            "stem": stem,
            "timesteps": 16,
            "soma_hardware_latency_s": float(soma["hardware_latency_s"]),
            "sana_hardware_latency_s": sana_latency,
            "hardware_latency_error_percent": relative_percent(
                float(soma["hardware_latency_s"]), sana_latency),
            "soma_energy_j": soma_energy,
            "sana_energy_j": sana_energy,
            "energy_error_percent": relative_percent(soma_energy, sana_energy),
            "soma_host_latency_s": float(soma["host_latency_s"]),
            "sana_host_runtime_sum_s": sana_host,
            "host_speedup_sana_over_soma": sana_host / float(soma["host_latency_s"]),
            "functional_metric": functional_metric,
            "reference_output_spikes": reference_output_spikes,
            "soma_output_spikes": actual_output_spikes if item["kind"] != "head" else None,
            "functional_max_abs": functional_max_abs,
            "functional_match": functional_match,
            "soma_prediction": soma["prediction"],
            "reference_prediction": item["reference_prediction"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"wrote {len(rows)} comparisons to {args.output}")


if __name__ == "__main__":
    main()

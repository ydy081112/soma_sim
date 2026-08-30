#!/usr/bin/env python3
"""把 CIFAR-10 单样本或已有 input_rates 编码成 SOMA-Sim spike CSV。"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


def load_cifar_pickle(path: Path, sample: int) -> Tuple[np.ndarray, int]:
    with path.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    image = np.asarray(batch[b"data"][sample], dtype=np.float32).reshape(3, 32, 32) / 255.0
    return image, int(batch[b"labels"][sample])


def load_rates(args: argparse.Namespace) -> Tuple[np.ndarray, int]:
    if args.sample_npz:
        sample = np.load(args.sample_npz, allow_pickle=False)
        if "input_rates" in sample:
            return np.asarray(sample["input_rates"], dtype=np.float32), int(sample["label"])
        image = np.asarray(sample["normalized_image"], dtype=np.float32)
        label = int(sample["label"])
    else:
        image, label = load_cifar_pickle(args.cifar_batch, args.sample_index)

    # ON/OFF 保留归一化后可能出现的负值；rate_scale 全部来自命令行配置。
    scaled = image * args.rate_scale
    rates = np.concatenate((np.clip(scaled, 0.0, 1.0), np.clip(-scaled, 0.0, 1.0)), axis=0)
    return rates, label


def encode(rates: np.ndarray, timesteps: int, mode: str, seed: int) -> np.ndarray:
    if mode == "stochastic":
        rng = np.random.default_rng(seed)
        return rng.random((timesteps,) + rates.shape) < rates[None, ...]
    # 可复现的 phase accumulator rate coding，不依赖随机数。
    phase = np.zeros_like(rates, dtype=np.float32)
    output = np.zeros((timesteps,) + rates.shape, dtype=np.bool_)
    for timestep in range(timesteps):
        phase += rates
        output[timestep] = phase >= 1.0
        phase[output[timestep]] -= 1.0
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cifar-batch", type=Path)
    source.add_argument("--sample-npz", type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--rate-scale", type=float, default=1.0)
    parser.add_argument("--mode", choices=("deterministic", "stochastic"), default="deterministic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-id", default="input")
    parser.add_argument("--src-pe", type=int, default=0)
    parser.add_argument("--src-router", type=int, default=0)
    parser.add_argument("--dst-pe", type=int, default=1)
    parser.add_argument("--dst-router", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("input/input_spike.csv"))
    args = parser.parse_args()
    rates, label = load_rates(args)
    spikes = encode(np.clip(rates, 0.0, 1.0), args.timesteps, args.mode, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["generated_time", "current_time", "spike_id", "timestep", "layer_id",
              "src_neuron", "src_pe", "src_router", "dst_pe", "dst_router", "route",
              "value", "expected_output"]
    spike_id = 0
    channels = spikes.shape[1]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for timestep_index in range(args.timesteps):
            timestep = timestep_index + 1
            # Simulator layout 是 spatial-major [H*W,C]。
            for channel, y, x in np.argwhere(spikes[timestep_index]):
                neuron = (int(y) * spikes.shape[3] + int(x)) * channels + int(channel)
                writer.writerow({
                    "generated_time": 0, "current_time": 0,
                    "spike_id": spike_id, "timestep": timestep, "layer_id": args.layer_id,
                    "src_neuron": neuron, "src_pe": args.src_pe, "src_router": args.src_router,
                    "dst_pe": args.dst_pe, "dst_router": args.dst_router,
                    "route": "debug_only", "value": 1.0, "expected_output": label,
                })
                spike_id += 1
    print(f"wrote {spike_id} spikes to {args.output}; label={label}")


if __name__ == "__main__":
    main()

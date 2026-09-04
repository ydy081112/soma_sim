#!/usr/bin/env python3
"""按单一 YAML 配置生成资产、运行 C++ SOMA-Sim 并核对 NIR reference。"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run(command):
    print("运行：", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    vit, architecture = config["vit"], config["architecture"]
    if vit["mapping_strategy"] != "adjacent_qkv_attention_then_projection_mlp":
        raise ValueError("当前工具不支持所选 mapping_strategy")
    nir = Path(vit["nir_path"])
    expected_block = f"block_{int(vit['block']):02d}"
    if expected_block not in nir.stem:
        raise ValueError("block 配置与 NIR 文件名不一致")
    npz = root / vit["runtime_npz"]
    spikes = root / vit["input_spikes"]
    mapping = root / vit["mapping"]
    manifest = root / vit["manifest"]
    output = root / vit["output"]
    run([sys.executable, root / "tools/prepare_vit_block00_assets.py",
         "--nir", nir, "--npz", npz, "--input-csv", spikes,
         "--manifest", manifest, "--mapping", mapping,
         "--input-tensor", vit["input_tensor"]])
    run([root / "build/soma-sim", "--hardware", root / architecture["base"],
         "--mapping", mapping, "--weights", npz, "--input", spikes,
         "--output", output])
    run([sys.executable, root / "tools/compare_vit_block00_reference.py",
         "--npz", npz, "--manifest", manifest,
         "--trace", output / "firing_trace.csv",
         "--json", output / "correctness.json",
         "--markdown", output / "correctness.md"])


if __name__ == "__main__":
    main()

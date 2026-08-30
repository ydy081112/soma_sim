#!/usr/bin/env python3
"""生成仓库最小样例的 Spatial Pattern Template/source-major NPZ。"""

from pathlib import Path
import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("input/weights.npz"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 1x1 Conv: [Cin, Kh, Kw, Cout]，每个输入空间点复用同一模板。
    arrays = {
        "conv0_weight": np.asarray([1.25, 0.25], dtype=np.float32).reshape(1, 1, 1, 2),
        "conv0_bias": np.zeros(2, dtype=np.float32),
        "conv0_plan_pattern_id": np.zeros(4, dtype=np.int32),
        "conv0_plan_dst_base": np.arange(4, dtype=np.int32),
        "conv0_pattern_ptr": np.asarray([0, 1], dtype=np.int32),
        "conv0_pattern_dst_offset": np.asarray([0], dtype=np.int32),
        "conv0_pattern_weight_offset": np.asarray([0], dtype=np.int64),
    }
    # [Cin, Cout]：conv 的偶数 neuron 发向 class 0，奇数 neuron 发向 class 1。
    dense = np.zeros((8, 2), dtype=np.float32)
    dense[0::2, 0] = 1.0
    dense[1::2, 1] = 1.0
    arrays["readout_weight"] = dense
    arrays["readout_bias"] = np.zeros(2, dtype=np.float32)
    np.savez(args.output, **arrays)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()


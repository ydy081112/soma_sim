#!/usr/bin/env python3
"""比较 SOMA-Sim summary 与保留的 SANA-FE VGG16 reference。"""

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("output_vgg16/summary.json"))
    parser.add_argument("--reference", type=Path,
                        default=Path("reference/sanafe_vgg16_timestep_metrics.npz"))
    parser.add_argument("--output", type=Path,
                        default=Path("output_vgg16/reference_comparison.json"))
    args = parser.parse_args()
    ours = json.loads(args.summary.read_text(encoding="utf-8"))
    reference = np.load(args.reference, allow_pickle=False)
    scores = np.asarray(ours["output_scores"], dtype=np.float64)
    expected = np.asarray(reference["final_logits"], dtype=np.float64)
    cosine = float(scores @ expected / (np.linalg.norm(scores) * np.linalg.norm(expected)))
    result = {
        "soma_completed": bool(ours["completed"]),
        "soma_prediction": int(ours["prediction"]),
        "reference_prediction": int(reference["prediction"]),
        "label": int(reference["label"]),
        "prediction_match": int(ours["prediction"]) == int(reference["prediction"]),
        "score_cosine_similarity": cosine,
        "note": "SOMA 使用 timestep buffer 聚合与统一 neuron processing；比较分类和 score 方向相似度。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

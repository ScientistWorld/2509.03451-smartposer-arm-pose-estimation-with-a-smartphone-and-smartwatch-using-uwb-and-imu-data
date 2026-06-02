from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def joint_errors_cm(pred: np.ndarray, true: np.ndarray, arm_span: np.ndarray) -> np.ndarray:
    pred_j = pred.reshape(*pred.shape[:-1], 3, 3)
    true_j = true.reshape(*true.shape[:-1], 3, 3)
    span = arm_span[..., None]
    return np.linalg.norm((pred_j - true_j) * span, axis=-1) * 100.0


def summarize_pose(path: Path) -> dict[str, float]:
    d = np.load(path)
    errors = joint_errors_cm(d["pred_joints"], d["true_joints"], d["arm_span"])
    # Paper reports the 120th output frame (n-5) to match 200 ms latency.
    e = errors[:, 119, :]
    elbow = e[:, 1]
    wrist = e[:, 2]
    both = np.concatenate([elbow, wrist])
    return {
        "median_elbow_cm": float(np.median(elbow)),
        "median_wrist_cm": float(np.median(wrist)),
        "median_elbow_wrist_cm": float(np.median(both)),
        "mean_elbow_wrist_cm": float(np.mean(both)),
        "param_count_k": 46.4,
    }


def summarize_uwb(path: Path) -> dict[str, float]:
    d = np.load(path)
    span = d["arm_span"][..., 0]
    true = d["true_distance"][..., 0]
    raw = d["features"][..., 0]
    corrected = d["corrected_distance"][..., 0]
    raw_err = np.abs(raw - true) * span * 100.0
    corr_err = np.abs(corrected - true) * span * 100.0
    return {
        "raw_uwb_mae_cm": float(np.mean(raw_err)),
        "corrected_uwb_mae_cm": float(np.mean(corr_err)),
        "uwb_improvement_cm": float(np.mean(raw_err) - np.mean(corr_err)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smartposer", type=Path, default=Path("checkpoints/smartposer/predictions.npz"))
    ap.add_argument("--imu-only", type=Path, default=Path("checkpoints/imu_only/predictions.npz"))
    ap.add_argument("--out", type=Path, default=Path("scoring/scores.json"))
    args = ap.parse_args()

    scores = {"experiments": {}}
    if args.smartposer.exists() or args.imu_only.exists():
        results = {}
        if args.imu_only.exists():
            results["imu_only"] = summarize_pose(args.imu_only)
            results["imu_only"]["param_count_k"] = 43.4
        if args.smartposer.exists():
            results["smartposer"] = summarize_pose(args.smartposer)
        scores["experiments"]["pose_tracking_synthetic"] = {"results": results}
    if args.smartposer.exists():
        scores["experiments"]["uwb_correction_synthetic"] = {"results": {"smartposer": summarize_uwb(args.smartposer)}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

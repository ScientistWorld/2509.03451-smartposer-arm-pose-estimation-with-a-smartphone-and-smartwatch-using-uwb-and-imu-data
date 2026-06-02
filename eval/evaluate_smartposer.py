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


def _paths_from_pattern(pattern: str) -> list[Path]:
    paths = sorted(Path().glob(pattern))
    if not paths and Path(pattern).exists():
        paths = [Path(pattern)]
    return paths


def _metadata_param_count_k(prediction_path: Path) -> float:
    metadata_path = prediction_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing required metadata file: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if "param_count_k" not in metadata:
        raise KeyError(f"metadata file {metadata_path} does not contain param_count_k")
    return float(metadata["param_count_k"])


def _load_concat(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise FileNotFoundError("No prediction artifacts matched the requested path/pattern")
    loaded = [np.load(path) for path in paths]
    keys = loaded[0].files
    return {key: np.concatenate([item[key] for item in loaded], axis=0) for key in keys}


def summarize_pose(paths: list[Path]) -> dict[str, float]:
    d = _load_concat(paths)
    errors = joint_errors_cm(d["pred_joints"], d["true_joints"], d["arm_span"])
    # Paper reports the 120th output frame (n-5) to match 200 ms latency.
    e = errors[:, 119, :]
    elbow = e[:, 1]
    wrist = e[:, 2]
    both = np.concatenate([elbow, wrist])
    true_j = d["true_joints"].reshape(*d["true_joints"].shape[:-1], 3, 3)
    pred_j = d["pred_joints"].reshape(*d["pred_joints"].shape[:-1], 3, 3)
    motion = np.linalg.norm(true_j[:, 119, 2, :] - true_j[:, 94, 2, :], axis=-1) * d["arm_span"][:, 119, 0]
    high_motion = motion >= np.quantile(motion, 0.75)
    per_window_both = np.stack([elbow, wrist], axis=1)
    pred_forearm = np.linalg.norm(pred_j[:, 119, 2, :] - pred_j[:, 119, 1, :], axis=-1)
    true_forearm = np.linalg.norm(true_j[:, 119, 2, :] - true_j[:, 119, 1, :], axis=-1)
    return {
        "median_elbow_cm": float(np.median(elbow)),
        "median_wrist_cm": float(np.median(wrist)),
        "median_elbow_wrist_cm": float(np.median(both)),
        "mean_elbow_wrist_cm": float(np.mean(both)),
        "p90_elbow_wrist_cm": float(np.percentile(both, 90)),
        "high_motion_mean_elbow_wrist_cm": float(np.mean(per_window_both[high_motion])),
        "forearm_length_mae_cm": float(np.mean(np.abs(pred_forearm - true_forearm) * d["arm_span"][:, 119, 0] * 100.0)),
        "param_count_k": _metadata_param_count_k(paths[0]),
    }


def summarize_uwb(paths: list[Path]) -> dict[str, float]:
    d = _load_concat(paths)
    span = d["arm_span"][..., 0]
    true = d["true_distance"][..., 0]
    raw = d["features"][..., 0]
    corrected = d["corrected_distance"][..., 0]
    raw_err = np.abs(raw - true) * span * 100.0
    corr_err = np.abs(corrected - true) * span * 100.0
    hard_mask = raw_err >= np.quantile(raw_err, 0.75)
    return {
        "raw_uwb_mae_cm": float(np.mean(raw_err)),
        "corrected_uwb_mae_cm": float(np.mean(corr_err)),
        "corrected_uwb_p90_cm": float(np.percentile(corr_err, 90)),
        "uwb_improvement_cm": float(np.mean(raw_err) - np.mean(corr_err)),
        "hard_uwb_improvement_cm": float(np.mean(raw_err[hard_mask]) - np.mean(corr_err[hard_mask])),
        "uwb_correction_win_rate": float(np.mean(corr_err < raw_err)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smartposer", default="checkpoints/smartposer*/predictions.npz")
    ap.add_argument("--imu-only", default="checkpoints/imu_only*/predictions.npz")
    ap.add_argument("--out", type=Path, default=Path("scoring/scores.json"))
    ap.add_argument("--allow-partial", action="store_true", help="Write scores for whichever artifacts exist instead of requiring both methods.")
    args = ap.parse_args()

    scores = {"experiments": {}}
    smartposer_paths = _paths_from_pattern(args.smartposer)
    imu_paths = _paths_from_pattern(args.imu_only)
    if not args.allow_partial:
        if not smartposer_paths:
            raise FileNotFoundError(f"Missing required SmartPoser artifacts for pattern: {args.smartposer}")
        if not imu_paths:
            raise FileNotFoundError(f"Missing required IMU-only artifacts for pattern: {args.imu_only}")

    results = {}
    if imu_paths:
        results["imu_only"] = summarize_pose(imu_paths)
    if smartposer_paths:
        results["smartposer"] = summarize_pose(smartposer_paths)
    if results:
        scores["experiments"]["pose_tracking_synthetic"] = {"results": results}
    if smartposer_paths:
        scores["experiments"]["uwb_correction_synthetic"] = {"results": {"smartposer": summarize_uwb(smartposer_paths)}}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

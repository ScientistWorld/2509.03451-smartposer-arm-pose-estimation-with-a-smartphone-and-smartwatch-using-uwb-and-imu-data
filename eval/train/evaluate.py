from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np


SLICE = "train"
DEFAULT_OUT = Path("scoring/scores_train.json")
DEFAULT_PATTERN = "checkpoints/*/predictions.npz"
POSE_KEYS = {"pred_joints", "true_joints", "arm_span"}
UWB_KEYS = {"features", "true_distance", "corrected_distance", "arm_span"}


def _paths_from_pattern(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern)]
    literal = Path(pattern)
    if literal.exists():
        paths.append(literal)
    return sorted(set(paths))


def _collect_prediction_paths(args: argparse.Namespace) -> list[Path]:
    patterns: list[str] = []
    if args.smartposer:
        patterns.append(args.smartposer)
    if args.imu_only:
        patterns.append(args.imu_only)
    if args.predictions:
        patterns.extend(args.predictions)
    if not patterns:
        patterns.append(DEFAULT_PATTERN)

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(_paths_from_pattern(pattern))
    paths = sorted(set(paths))
    if not paths and not args.allow_partial:
        raise FileNotFoundError(f"No prediction artifacts matched: {patterns}")
    return paths


def _metadata(path: Path) -> dict:
    metadata_path = path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text())


def _method_name(path: Path) -> str:
    metadata = _metadata(path)
    method = metadata.get("method")
    if isinstance(method, str) and method:
        return method
    return re.sub(r"[_-]?fold[_-]?\d+$", "", path.parent.name)


def _group_by_method(paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(_method_name(path), []).append(path)
    return {method: sorted(method_paths) for method, method_paths in sorted(grouped.items())}


def _metadata_param_count_k(paths: list[Path]) -> float:
    counts = []
    for path in paths:
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing required metadata file: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if "param_count_k" not in metadata:
            raise KeyError(f"metadata file {metadata_path} does not contain param_count_k")
        counts.append(float(metadata["param_count_k"]))
    return float(max(counts))


def _slice_mask(n: int, block_size: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=bool)
    idx = np.arange(n)
    mask = ((idx // block_size) % 2) == 0
    if mask.sum() == 0 or mask.sum() == n:
        midpoint = n // 2
        mask = idx < midpoint
    return mask


def _load_concat(paths: list[Path], required_keys: set[str], block_size: int) -> dict[str, np.ndarray]:
    if not paths:
        raise FileNotFoundError("No prediction artifacts were provided")
    arrays = {key: [] for key in sorted(required_keys)}
    for path in paths:
        with np.load(path) as data:
            missing = required_keys - set(data.files)
            if missing:
                raise KeyError(f"{path} is missing required keys for {SLICE} scoring: {sorted(missing)}")
            first_key = next(iter(required_keys))
            mask = _slice_mask(int(data[first_key].shape[0]), block_size)
            for key in arrays:
                arrays[key].append(data[key][mask])
    return {key: np.concatenate(chunks, axis=0) for key, chunks in arrays.items()}


def _all_paths_have_keys(paths: list[Path], required_keys: set[str]) -> bool:
    for path in paths:
        with np.load(path) as data:
            if required_keys - set(data.files):
                return False
    return True


def joint_errors_cm(pred: np.ndarray, true: np.ndarray, arm_span: np.ndarray) -> np.ndarray:
    pred_j = pred.reshape(*pred.shape[:-1], 3, 3)
    true_j = true.reshape(*true.shape[:-1], 3, 3)
    span = arm_span[..., None]
    return np.linalg.norm((pred_j - true_j) * span, axis=-1) * 100.0


def _elbow_wrist_errors_at_target(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    errors = joint_errors_cm(d["pred_joints"], d["true_joints"], d["arm_span"])
    target = errors[:, 119, :]
    elbow = target[:, 1]
    wrist = target[:, 2]
    return elbow, wrist, np.concatenate([elbow, wrist])


def bone_length_mae_cm(pred: np.ndarray, true: np.ndarray, arm_span: np.ndarray) -> float:
    pred_j = pred.reshape(*pred.shape[:-1], 3, 3)
    true_j = true.reshape(*true.shape[:-1], 3, 3)
    pred_len = np.linalg.norm(pred_j[:, 119, 2] - pred_j[:, 119, 1], axis=-1)
    true_len = np.linalg.norm(true_j[:, 119, 2] - true_j[:, 119, 1], axis=-1)
    span = arm_span[:, 119, 0]
    return float(np.mean(np.abs(pred_len - true_len) * span * 100.0))


def high_motion_mean_cm(d: dict[str, np.ndarray], both_errors: np.ndarray) -> float:
    true_j = d["true_joints"].reshape(*d["true_joints"].shape[:-1], 3, 3)
    wrist = true_j[:, :, 2, :]
    motion = np.linalg.norm(wrist[:, 119, :] - wrist[:, 94, :], axis=-1) * d["arm_span"][:, 119, 0]
    if motion.size == 0:
        return float("nan")
    threshold = np.quantile(motion, 0.75)
    mask = motion >= threshold
    sample_errors = both_errors.reshape(2, -1).T
    return float(np.mean(sample_errors[mask]))


def summarize_pose(paths: list[Path], block_size: int) -> dict[str, float]:
    d = _load_concat(paths, POSE_KEYS, block_size)
    elbow, wrist, both = _elbow_wrist_errors_at_target(d)
    return {
        "median_elbow_cm": float(np.median(elbow)),
        "median_wrist_cm": float(np.median(wrist)),
        "median_elbow_wrist_cm": float(np.median(both)),
        "mean_elbow_wrist_cm": float(np.mean(both)),
        "p90_elbow_wrist_cm": float(np.percentile(both, 90)),
        "high_motion_mean_elbow_wrist_cm": high_motion_mean_cm(d, both),
        "forearm_length_mae_cm": bone_length_mae_cm(d["pred_joints"], d["true_joints"], d["arm_span"]),
        "param_count_k": _metadata_param_count_k(paths),
    }


def summarize_uwb(paths: list[Path], block_size: int) -> dict[str, float]:
    d = _load_concat(paths, UWB_KEYS, block_size)
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


def build_scores(paths: list[Path], block_size: int) -> dict:
    scores = {"slice": SLICE, "experiments": {}}
    grouped = _group_by_method(paths)

    pose_results = {}
    uwb_results = {}
    for method, method_paths in grouped.items():
        if _all_paths_have_keys(method_paths, POSE_KEYS):
            pose_results[method] = summarize_pose(method_paths, block_size)
        if _all_paths_have_keys(method_paths, UWB_KEYS):
            uwb_results[method] = summarize_uwb(method_paths, block_size)

    if pose_results:
        scores["experiments"]["pose_tracking_synthetic"] = {"results": pose_results}
    if uwb_results:
        scores["experiments"]["uwb_correction_synthetic"] = {"results": uwb_results}
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", action="append", help="Prediction artifact glob; may be supplied multiple times.")
    ap.add_argument("--smartposer", default=None, help="Legacy SmartPoser prediction glob.")
    ap.add_argument("--imu-only", dest="imu_only", default=None, help="Legacy IMU-only prediction glob.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--block-size", type=int, default=64, help="Temporal block size for the deterministic visible slice.")
    ap.add_argument("--allow-partial", action="store_true", help="Write scores for whichever artifacts exist.")
    args = ap.parse_args()

    if args.block_size < 1:
        raise ValueError("--block-size must be positive")
    paths = _collect_prediction_paths(args)
    scores = build_scores(paths, args.block_size)
    if not scores["experiments"] and not args.allow_partial:
        raise RuntimeError("No scoreable prediction artifacts were found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()

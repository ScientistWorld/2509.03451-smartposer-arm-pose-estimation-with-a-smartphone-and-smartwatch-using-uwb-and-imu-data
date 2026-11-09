"""Visible (development) evaluator for the phone/watch arm-tracking gym.

Leave-one-participant-out over the 10 visible development participants in
``data/smartposer_dev``: the fold-k model of every submission
(``checkpoints/<method>/fold_<k>/model.pt``, TorchScript) is run on eval-built
rolling windows of participant k, which that model must not have trained on.
Scores are computed here from the model outputs and the development ground
truth; nothing else a submission writes is read. Gates mirror the hidden test:
a submission that fails to load/run, exceeds 92.8k parameters (or a ~1 MB
uncompressed archive), returns
non-finite or mis-shaped outputs, does not beat the best constant pose, or does
not beat the raw range stream is scored at the no-skill floor.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SLICE = "train"
DEFAULT_DATA = Path("data/smartposer_dev")
DEFAULT_CHECKPOINTS = Path("checkpoints")
DEFAULT_OUT = Path("scoring/scores_train.json")
FIXED_DIR = HERE / "fixed_models"

WINDOW = 125
TARGET_INDEX = 119  # 120th output frame: 5 frames (200 ms) of look-ahead
LOOK_AHEAD = WINDOW - 1 - TARGET_INDEX
STRIDE = 5
MOTION_LAG = 25
N_FOLDS = 10
BATCH = 512
ORDER_SEED = 1109
PARAM_LIMIT = 92_800
STORAGE_LIMIT_BYTES = 8 * PARAM_LIMIT
ARCHIVE_LIMIT_BYTES = STORAGE_LIMIT_BYTES + 256 * 1024  # tensors + code/pickles/extra files
STORAGE_PATTERN = re.compile(r"^[^/]+/(?:data|constants)/[^/]+$")

POSE_EXP = "pose_tracking_synthetic"
RANGE_EXP = "uwb_correction_synthetic"
POSE_FLOOR = "mean_pose"
RANGE_FLOOR = "raw_uwb"
FIXED_POSE_BASELINES = ("imu_only",)
RESERVED = {POSE_FLOOR, RANGE_FLOOR, *FIXED_POSE_BASELINES}


class InvalidSubmission(Exception):
    pass


def load_dev(data_dir: Path) -> list[dict[str, np.ndarray]]:
    folds = []
    for k in range(N_FOLDS):
        path = data_dir / f"subject_{k:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing development participant {path}; run scripts/download.sh")
        with np.load(path) as d:
            feats = np.ascontiguousarray(d["features"], dtype=np.float32)
            joints = d["joints"].astype(np.float64)
            dist = d["true_distance"][:, 0].astype(np.float64)
            span = float(d["arm_span"])
        frames = feats.shape[0]
        targets = np.arange(TARGET_INDEX, frames - LOOK_AHEAD, STRIDE)
        view = np.lib.stride_tricks.sliding_window_view(feats, WINDOW, axis=0)
        folds.append(
            {
                "X": np.ascontiguousarray(view[targets - TARGET_INDEX].transpose(0, 2, 1)),
                "joints": joints[targets],
                "dist": dist[targets],
                "raw": feats[targets, 0].astype(np.float64),
                "span": np.full(targets.size, span),
                "motion": np.linalg.norm(joints[targets, 6:9] - joints[targets - MOTION_LAG, 6:9], axis=1) * span,
            }
        )
    return folds


def pooled_labels(folds: list[dict]) -> dict[str, np.ndarray]:
    pooled = {key: np.concatenate([f[key] for f in folds]) for key in ("joints", "dist", "raw", "span", "motion")}
    pooled["high_motion"] = pooled["motion"] >= np.quantile(pooled["motion"], 0.75)
    raw_err = np.abs(pooled["raw"] - pooled["dist"]) * pooled["span"]
    pooled["hard_range"] = raw_err >= np.quantile(raw_err, 0.75)
    return pooled


def pose_metrics(pred: np.ndarray, true: np.ndarray, span: np.ndarray, high_motion: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm((pred - true).reshape(-1, 3, 3), axis=-1) * span[:, None] * 100.0
    elbow, wrist = err[:, 1], err[:, 2]
    both = np.concatenate([elbow, wrist])
    per_frame = 0.5 * (elbow + wrist)
    return {
        "median_elbow_cm": float(np.median(elbow)),
        "median_wrist_cm": float(np.median(wrist)),
        "median_elbow_wrist_cm": float(np.median(both)),
        "mean_elbow_wrist_cm": float(np.mean(both)),
        "p90_elbow_wrist_cm": float(np.percentile(both, 90)),
        "high_motion_mean_elbow_wrist_cm": float(np.mean(per_frame[high_motion])),
    }


def range_metrics(pred: np.ndarray, true: np.ndarray, raw: np.ndarray, span: np.ndarray, hard: np.ndarray) -> dict[str, float]:
    corrected_err = np.abs(pred - true) * span * 100.0
    raw_err = np.abs(raw - true) * span * 100.0
    mae = float(np.mean(corrected_err))
    return {
        "uwb_error_reduction_pct": 100.0 * (1.0 - mae / float(np.mean(raw_err))),
        "corrected_uwb_mae_cm": mae,
        "corrected_uwb_p90_cm": float(np.percentile(corrected_err, 90)),
        "occluded_uwb_mae_cm": float(np.mean(corrected_err[hard])),
        "uwb_correction_win_rate": float(np.mean(corrected_err < raw_err)),
    }


def model_size(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise InvalidSubmission(f"missing {path}")
    try:
        with zipfile.ZipFile(path) as zf:
            entries = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise InvalidSubmission(f"{path} is not a TorchScript archive") from exc
    archive = sum(e.file_size for e in entries)
    if archive > ARCHIVE_LIMIT_BYTES:
        raise InvalidSubmission(f"archive holds {archive} bytes uncompressed, above the {ARCHIVE_LIMIT_BYTES}-byte limit")
    storage = sum(e.file_size for e in entries if STORAGE_PATTERN.match(e.filename))
    module = torch.jit.load(str(path), map_location="cpu")
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        key = (tensor.data_ptr(), tensor.numel())
        if key not in seen:
            seen.add(key)
            total += int(tensor.numel())
    return total, int(storage)


def _validated(out, batch: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise InvalidSubmission("forward must return a (range, joints) pair")
    range_t, joints_t = out
    if not isinstance(range_t, torch.Tensor) or not isinstance(joints_t, torch.Tensor):
        raise InvalidSubmission("forward outputs must be tensors")
    if tuple(range_t.shape) != (batch, WINDOW, 1) or tuple(joints_t.shape) != (batch, WINDOW, 9):
        raise InvalidSubmission(
            f"output shapes {tuple(range_t.shape)} / {tuple(joints_t.shape)}; expected ({batch}, {WINDOW}, 1) / ({batch}, {WINDOW}, 9)"
        )
    r = range_t[:, TARGET_INDEX, 0].detach().double().cpu().numpy()
    j = joints_t[:, TARGET_INDEX, :].detach().double().cpu().numpy()
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(j))):
        raise InvalidSubmission("non-finite outputs")
    return r, j


def predict(path: Path, X: np.ndarray, device: torch.device, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    range_out = np.empty(n)
    pose_out = np.empty((n, 9))
    for start in range(0, n, BATCH):
        idx = order[start : start + BATCH]
        module = torch.jit.load(str(path), map_location=device)
        module.eval()
        with torch.inference_mode():
            out = module(torch.from_numpy(X[idx]).to(device))
        r, j = _validated(out, len(idx))
        range_out[idx] = r
        pose_out[idx] = j
    return range_out, pose_out


def score_submission(fold_paths: dict[int, Path], folds: list[dict], labels: dict, device: torch.device) -> dict:
    status: dict = {"valid": True, "reasons": [], "param_count": None, "storage_bytes": None}
    ranges, poses, counts, storages = [], [], [], []
    for k in range(N_FOLDS):
        path = fold_paths.get(k)
        try:
            if path is None:
                raise InvalidSubmission("missing fold model")
            count, storage = model_size(path)
            counts.append(count)
            storages.append(storage)
            if count > PARAM_LIMIT:
                raise InvalidSubmission(f"{count} parameters+buffers exceed the {PARAM_LIMIT} limit")
            if storage > STORAGE_LIMIT_BYTES:
                raise InvalidSubmission(f"{storage} bytes of stored tensors exceed the {STORAGE_LIMIT_BYTES}-byte limit")
            order = np.random.default_rng(ORDER_SEED + k).permutation(folds[k]["X"].shape[0])
            r, j = predict(path, folds[k]["X"], device, order)
        except Exception as exc:
            status["valid"] = False
            status["reasons"].append(f"fold {k}: {type(exc).__name__}: {exc}"[:500])
            break
        ranges.append(r)
        poses.append(j)
    if counts:
        status["param_count"] = int(max(counts))
        status["storage_bytes"] = int(max(storages))
    if not status["valid"]:
        return status
    status["pose"] = pose_metrics(np.concatenate(poses), labels["joints"], labels["span"], labels["high_motion"])
    status["range"] = range_metrics(np.concatenate(ranges), labels["dist"], labels["raw"], labels["span"], labels["hard_range"])
    return status


def find_submissions(checkpoints: Path) -> tuple[dict[str, dict[int, Path]], list[str]]:
    found: dict[str, dict[int, Path]] = {}
    skipped: list[str] = []
    if not checkpoints.is_dir():
        return found, skipped
    for d in sorted(p for p in checkpoints.iterdir() if p.is_dir()):
        if not any(p.is_dir() for p in d.glob("fold_*")):
            skipped.append(f"{d.name}: no fold_<k>/model.pt submission layout")
            continue
        if d.name in RESERVED or d.name.startswith("_"):
            skipped.append(f"{d.name}: reserved name (fixed by eval/)")
            continue
        found[d.name] = {k: d / f"fold_{k}" / "model.pt" for k in range(N_FOLDS) if (d / f"fold_{k}" / "model.pt").exists()}
    return found, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    folds = load_dev(args.data_dir)
    labels = pooled_labels(folds)
    floor_pose = pose_metrics(
        np.broadcast_to(labels["joints"].mean(axis=0), labels["joints"].shape), labels["joints"], labels["span"], labels["high_motion"]
    )
    floor_pose["param_count_k"] = 0.009
    floor_range = range_metrics(labels["raw"], labels["dist"], labels["raw"], labels["span"], labels["hard_range"])

    pose_results = {POSE_FLOOR: dict(floor_pose)}
    range_results = {RANGE_FLOOR: dict(floor_range)}
    statuses: dict[str, dict] = {}
    skipped: list[str] = []

    for name in FIXED_POSE_BASELINES:
        fold_paths = {k: FIXED_DIR / name / f"fold_{k}.pt" for k in range(N_FOLDS)}
        if not all(p.exists() for p in fold_paths.values()):
            skipped.append(f"{name}: fixed baseline models not installed under {FIXED_DIR / name}")
            continue
        st = score_submission(fold_paths, folds, labels, device)
        st.pop("range", None)
        if st["valid"] and st["pose"]["median_elbow_wrist_cm"] >= floor_pose["median_elbow_wrist_cm"]:
            st["valid"] = False
            st["reasons"].append("pose error not below the constant-pose floor")
        row = dict(st["pose"]) if st["valid"] else dict(floor_pose)
        row["param_count_k"] = (st["param_count"] or 0) / 1000.0
        pose_results[name] = row
        statuses[name] = st

    submissions, found_skipped = find_submissions(args.checkpoints_dir)
    skipped.extend(found_skipped)
    for name, fold_paths in submissions.items():
        st = score_submission(fold_paths, folds, labels, device)
        if st["valid"]:
            if st["pose"]["median_elbow_wrist_cm"] >= floor_pose["median_elbow_wrist_cm"]:
                st["valid"] = False
                st["reasons"].append("pose error not below the constant-pose floor (mean_pose)")
            if st["range"]["corrected_uwb_mae_cm"] >= floor_range["corrected_uwb_mae_cm"]:
                st["valid"] = False
                st["reasons"].append("range error not below the raw UWB stream (raw_uwb)")
        pose_row = dict(st["pose"]) if st["valid"] else dict(floor_pose)
        pose_row["param_count_k"] = (st["param_count"] or 0) / 1000.0
        pose_results[name] = pose_row
        range_results[name] = dict(st["range"]) if st["valid"] else dict(floor_range)
        statuses[name] = st
        print(json.dumps({"method": name, "valid": st["valid"], "reasons": st["reasons"]}), flush=True)

    scores = {
        "slice": SLICE,
        "experiments": {
            POSE_EXP: {"results": {k: pose_results[k] for k in sorted(pose_results)}},
            RANGE_EXP: {"results": {k: range_results[k] for k in sorted(range_results)}},
        },
        "_status": statuses,
        "_meta": {
            "participants": N_FOLDS,
            "windows": int(sum(f["X"].shape[0] for f in folds)),
            "target_index": TARGET_INDEX,
            "stride": STRIDE,
            "param_limit": PARAM_LIMIT,
            "storage_limit_bytes": STORAGE_LIMIT_BYTES,
            "archive_limit_bytes": ARCHIVE_LIMIT_BYTES,
            "device": str(device),
            "skipped": skipped,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores["experiments"], indent=2))


if __name__ == "__main__":
    main()

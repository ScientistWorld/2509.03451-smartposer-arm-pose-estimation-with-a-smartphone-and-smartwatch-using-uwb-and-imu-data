"""Hidden-test evaluator for the phone/watch arm-tracking gym.

The held-out participants are generated here, at test time, from a private
seed; their sensor streams and ground truth never exist outside this process.
Every submission under ``checkpoints/<method>/fold_<k>/model.pt`` (TorchScript,
one model per leave-one-participant-out fold) is run on eval-built rolling
windows of every held-out participant, and every scored number is computed
here from the model outputs and the private ground truth. No number, flag,
label, comparator or metadata file written by a submission is read.

Gates (any failure scores the submission at the no-skill floor on every metric):
  * every directory under ``checkpoints/`` whose name does not start with ``_``
    (and is not a reserved baseline name) is a submission and must hold all
    10 fold models;
  * every fold model loads as TorchScript, holds at most 92.8k parameters and
    buffers (and at most the matching tensor storage), keeps the whole archive
    (code, pickled attributes, extra files) under ~1 MB uncompressed, and
    returns finite ``(range [B,125,1], joints [B,125,9])`` for every window;
  * each window's output does not depend on which other windows share its
    batch (a re-batched probe must reproduce the outputs to 1e-3 arm spans);
  * pooled median elbow/wrist error beats the best constant pose (mean_pose);
  * mean range error beats the raw UWB stream (raw_uwb).

A valid submission's scored metrics are clipped at the floor values, so a
failed submission always holds the worst value of every scored metric.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import frozen_simulator as sim  # noqa: E402

SLICE = "test"
DEFAULT_CHECKPOINTS = Path("checkpoints")
DEFAULT_OUT = Path("scoring/scores_test.json")
FINGERPRINT = HERE / "fingerprint.json"
FIXED_DIR = HERE / "fixed_models"

TEST_SEED = 0x6A1F_3C59_D2E8_47B1_9E04_5A7C_B3D1_28F6
TEST_SUBJECTS = 48
TEST_FRAMES = 3000
WINDOW = 125
TARGET_INDEX = 119  # 120th output frame: 5 frames (200 ms) of look-ahead
LOOK_AHEAD = WINDOW - 1 - TARGET_INDEX
STRIDE = 5
MOTION_LAG = 25  # frames (1 s) for the high-motion slice
N_FOLDS = 10
BATCH = 512
ORDER_SEED = 20251109
PARAM_LIMIT = 92_800  # 2x the paper's 46.4k-parameter model
STORAGE_LIMIT_BYTES = 8 * PARAM_LIMIT
# Whole archive, uncompressed: tensor storage plus 256 KB for code, pickled
# attributes and extra files (the reference archive carries ~53 KB of these).
# Without it, capacity could hide in non-tensor attributes or code literals.
ARCHIVE_LIMIT_BYTES = STORAGE_LIMIT_BYTES + 256 * 1024
STORAGE_RE = re.compile(r"^[^/]+/(?:data|constants)/[^/]+$")
PROBE_WINDOWS = 64  # re-batched windows per fold model for the batch-independence gate
PROBE_TOL = 1e-3  # arm-span units (~0.17 cm); fp32 re-batching noise is ~1e-6

# Full-precision fp32 on GPU (no TF32), so scores and the probe are reproducible.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

POSE_EXP = "pose_tracking_synthetic"
RANGE_EXP = "uwb_correction_synthetic"
POSE_FLOOR = "mean_pose"
RANGE_FLOOR = "raw_uwb"
FIXED_POSE_BASELINES = ("imu_only",)
RESERVED = {POSE_FLOOR, RANGE_FLOOR, *FIXED_POSE_BASELINES}


class SubmissionError(Exception):
    pass


# ── Hidden participants ─────────────────────────────────────────────────────


def _fingerprint(subjects: list[dict]) -> list[list[float]]:
    return [
        [
            float(s["arm_span"]),
            float(np.mean(s["joints"], dtype=np.float64)),
            float(np.std(s["joints"], dtype=np.float64)),
            float(np.mean(s["true_distance"], dtype=np.float64)),
            float(np.mean(s["features"][:, 0], dtype=np.float64)),
            float(np.mean(s["features"][:, 22:25], dtype=np.float64)),
        ]
        for s in subjects
    ]


def _check_fingerprint(subjects: list[dict]) -> None:
    current = _fingerprint(subjects)
    if not FINGERPRINT.exists():
        FINGERPRINT.write_text(json.dumps({"simulator": sim.SIMULATOR_VERSION, "subjects": current}, indent=2) + "\n")
        return
    stored = json.loads(FINGERPRINT.read_text())
    if stored.get("simulator") != sim.SIMULATOR_VERSION or not np.allclose(
        np.asarray(stored["subjects"]), np.asarray(current), rtol=0.0, atol=1e-5
    ):
        raise RuntimeError(
            "Hidden test participants do not match eval/test/fingerprint.json; the numeric "
            "environment changed, so scores would not be comparable with the stored reference."
        )


def build_test_set() -> dict[str, np.ndarray]:
    subjects = sim.make_participants(TEST_SEED, TEST_SUBJECTS, TEST_FRAMES)
    _check_fingerprint(subjects)
    xs, joints, dist, raw, span, motion, who = [], [], [], [], [], [], []
    for pid, s in enumerate(subjects):
        feats = np.ascontiguousarray(s["features"], dtype=np.float32)
        frames = feats.shape[0]
        targets = np.arange(TARGET_INDEX, frames - LOOK_AHEAD, STRIDE)
        view = np.lib.stride_tricks.sliding_window_view(feats, WINDOW, axis=0)
        xs.append(np.ascontiguousarray(view[targets - TARGET_INDEX].transpose(0, 2, 1)))
        j = s["joints"].astype(np.float64)
        a = float(s["arm_span"])
        joints.append(j[targets])
        dist.append(s["true_distance"][targets, 0].astype(np.float64))
        raw.append(feats[targets, 0].astype(np.float64))
        span.append(np.full(targets.size, a))
        motion.append(np.linalg.norm(j[targets, 6:9] - j[targets - MOTION_LAG, 6:9], axis=1) * a)
        who.append(np.full(targets.size, pid))
    data = {
        "X": np.concatenate(xs).astype(np.float32),
        "joints": np.concatenate(joints),
        "dist": np.concatenate(dist),
        "raw": np.concatenate(raw),
        "span": np.concatenate(span),
        "motion": np.concatenate(motion),
        "participant": np.concatenate(who),  # not scored; for paired noise analysis
    }
    data["high_motion"] = data["motion"] >= np.quantile(data["motion"], 0.75)
    raw_err = np.abs(data["raw"] - data["dist"]) * data["span"]
    data["hard_range"] = raw_err >= np.quantile(raw_err, 0.75)
    return data


# ── Metrics ─────────────────────────────────────────────────────────────────


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
    corrected_mae = float(np.mean(corrected_err))
    return {
        "uwb_error_reduction_pct": 100.0 * (1.0 - corrected_mae / float(np.mean(raw_err))),
        "corrected_uwb_mae_cm": corrected_mae,
        "corrected_uwb_p90_cm": float(np.percentile(corrected_err, 90)),
        "occluded_uwb_mae_cm": float(np.mean(corrected_err[hard])),
        "uwb_correction_win_rate": float(np.mean(corrected_err < raw_err)),
    }


HIGHER_IS_BETTER = {"uwb_error_reduction_pct", "uwb_correction_win_rate"}


def clip_to_floor(metrics: dict[str, float], floor: dict[str, float]) -> dict[str, float]:
    """No scored metric of a valid submission may be worse than the no-skill floor."""
    return {k: (max(v, floor[k]) if k in HIGHER_IS_BETTER else min(v, floor[k])) for k, v in metrics.items()}


def pose_row(status: dict, floor_pose: dict[str, float]) -> dict[str, float]:
    scored_floor = {k: v for k, v in floor_pose.items() if k != "param_count_k"}
    if status["valid"]:
        row = clip_to_floor(status["pose"], scored_floor)
        row["param_count_k"] = status["param_count"] / 1000.0
    else:  # failed: floor on every scored metric, and at least the size limit
        row = dict(scored_floor)
        row["param_count_k"] = max(status["param_count"] or 0, PARAM_LIMIT) / 1000.0
    return row


# ── Running submissions ─────────────────────────────────────────────────────


def inspect_model(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise SubmissionError(f"missing {path}")
    try:
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        raise SubmissionError(f"{path} is not a TorchScript archive") from exc
    total = sum(info.file_size for info in infos)
    if total > ARCHIVE_LIMIT_BYTES:
        raise SubmissionError(f"archive holds {total} bytes uncompressed, above the {ARCHIVE_LIMIT_BYTES}-byte limit")
    storage = sum(info.file_size for info in infos if STORAGE_RE.match(info.filename))
    module = torch.jit.load(str(path), map_location="cpu")
    seen: set[tuple[int, int]] = set()
    count = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        key = (tensor.data_ptr(), tensor.numel())
        if key in seen:
            continue
        seen.add(key)
        count += int(tensor.numel())
    return count, int(storage)


def _check_output(out, batch: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise SubmissionError("forward must return a (range, joints) pair")
    rng_t, joints_t = out
    if not isinstance(rng_t, torch.Tensor) or not isinstance(joints_t, torch.Tensor):
        raise SubmissionError("forward outputs must be tensors")
    if tuple(rng_t.shape) != (batch, WINDOW, 1) or tuple(joints_t.shape) != (batch, WINDOW, 9):
        raise SubmissionError(
            f"output shapes {tuple(rng_t.shape)} / {tuple(joints_t.shape)}; expected ({batch}, {WINDOW}, 1) / ({batch}, {WINDOW}, 9)"
        )
    r = rng_t[:, TARGET_INDEX, 0].detach().double().cpu().numpy()
    j = joints_t[:, TARGET_INDEX, :].detach().double().cpu().numpy()
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(j))):
        raise SubmissionError("non-finite outputs")
    return r, j


def run_model(path: Path, X: np.ndarray, device: torch.device, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    range_out = np.empty(n)
    pose_out = np.empty((n, 9))
    for start in range(0, n, BATCH):
        idx = order[start : start + BATCH]
        # Fresh module per batch, windows shuffled across participants and time:
        # no state or batch statistics can carry information between windows.
        module = torch.jit.load(str(path), map_location=device)
        module.eval()
        with torch.inference_mode():
            out = module(torch.from_numpy(X[idx]).to(device))
        r, j = _check_output(out, len(idx))
        range_out[idx] = r
        pose_out[idx] = j
    return range_out, pose_out


def check_batch_independence(
    path: Path, X: np.ndarray, device: torch.device, order: np.ndarray, range_out: np.ndarray, pose_out: np.ndarray
) -> None:
    """Re-run windows from the first and the last batch together, in reverse
    order and at a different batch size; each window's output must not change.
    A model that pools information across the windows of a batch fails here."""
    half = PROBE_WINDOWS // 2
    idx = np.concatenate([order[:half], order[-half:]])[::-1].copy()
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    with torch.inference_mode():
        out = module(torch.from_numpy(X[idx]).to(device))
    r, j = _check_output(out, len(idx))
    gap = max(float(np.max(np.abs(r - range_out[idx]))), float(np.max(np.abs(j - pose_out[idx]))))
    if gap > PROBE_TOL:
        raise SubmissionError(f"outputs depend on the other windows in the batch (re-batched gap {gap:.3g} > {PROBE_TOL})")


def evaluate_submission(fold_paths: dict[int, Path], data: dict, device: torch.device, order: np.ndarray) -> dict:
    status: dict = {"valid": True, "reasons": [], "param_count": None, "storage_bytes": None}
    ranges, poses = [], []
    counts, storages = [], []
    for k in range(N_FOLDS):
        path = fold_paths.get(k)
        try:
            if path is None:
                raise SubmissionError("missing fold model")
            count, storage = inspect_model(path)
            counts.append(count)
            storages.append(storage)
            if count > PARAM_LIMIT:
                raise SubmissionError(f"{count} parameters+buffers exceed the {PARAM_LIMIT} limit")
            if storage > STORAGE_LIMIT_BYTES:
                raise SubmissionError(f"{storage} bytes of stored tensors exceed the {STORAGE_LIMIT_BYTES}-byte limit")
            r, j = run_model(path, data["X"], device, order)
            check_batch_independence(path, data["X"], device, order, r, j)
        except Exception as exc:  # any failure invalidates the whole submission
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
    reps = len(poses)
    status["pose"] = pose_metrics(
        np.concatenate(poses), np.tile(data["joints"], (reps, 1)), np.tile(data["span"], reps), np.tile(data["high_motion"], reps)
    )
    status["range"] = range_metrics(
        np.concatenate(ranges),
        np.tile(data["dist"], reps),
        np.tile(data["raw"], reps),
        np.tile(data["span"], reps),
        np.tile(data["hard_range"], reps),
    )
    return status


def discover(checkpoints: Path) -> tuple[dict[str, dict[int, Path]], list[str]]:
    submissions: dict[str, dict[int, Path]] = {}
    skipped: list[str] = []
    if not checkpoints.is_dir():
        return submissions, skipped
    for d in sorted(p for p in checkpoints.iterdir() if p.is_dir()):
        if d.name.startswith("_") or d.name.startswith("."):
            skipped.append(f"{d.name}: underscore/dot directory (not a submission)")
            continue
        if d.name in RESERVED:
            skipped.append(f"{d.name}: reserved name (fixed by eval/)")
            continue
        # Every other directory is a submission; missing folds fail it (scored at the floor).
        submissions[d.name] = {k: d / f"fold_{k}" / "model.pt" for k in range(N_FOLDS) if (d / f"fold_{k}" / "model.pt").exists()}
    return submissions, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    data = build_test_set()
    order = np.random.default_rng(ORDER_SEED).permutation(data["X"].shape[0])

    floor_pose = pose_metrics(
        np.broadcast_to(data["joints"].mean(axis=0), data["joints"].shape), data["joints"], data["span"], data["high_motion"]
    )
    floor_pose["param_count_k"] = 0.009
    floor_range = range_metrics(data["raw"], data["dist"], data["raw"], data["span"], data["hard_range"])

    pose_results = {POSE_FLOOR: dict(floor_pose)}
    range_results = {RANGE_FLOOR: dict(floor_range)}
    statuses: dict[str, dict] = {}
    skipped: list[str] = []

    for name in FIXED_POSE_BASELINES:
        fold_paths = {k: FIXED_DIR / name / f"fold_{k}.pt" for k in range(N_FOLDS)}
        if not all(p.exists() for p in fold_paths.values()):
            skipped.append(f"{name}: fixed baseline models not installed under {FIXED_DIR / name}")
            continue
        st = evaluate_submission(fold_paths, data, device, order)
        st.pop("range", None)
        if st["valid"] and st["pose"]["median_elbow_wrist_cm"] >= floor_pose["median_elbow_wrist_cm"]:
            st["valid"] = False
            st["reasons"].append("pose error not below the constant-pose floor")
        pose_results[name] = pose_row(st, floor_pose)
        statuses[name] = st

    submissions, found_skipped = discover(args.checkpoints_dir)
    skipped.extend(found_skipped)
    for name, fold_paths in submissions.items():
        st = evaluate_submission(fold_paths, data, device, order)
        if st["valid"]:
            if st["pose"]["median_elbow_wrist_cm"] >= floor_pose["median_elbow_wrist_cm"]:
                st["valid"] = False
                st["reasons"].append("pose error not below the constant-pose floor (mean_pose)")
            if st["range"]["corrected_uwb_mae_cm"] >= floor_range["corrected_uwb_mae_cm"]:
                st["valid"] = False
                st["reasons"].append("range error not below the raw UWB stream (raw_uwb)")
        pose_results[name] = pose_row(st, floor_pose)
        range_results[name] = clip_to_floor(st["range"], floor_range) if st["valid"] else dict(floor_range)
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
            "simulator": sim.SIMULATOR_VERSION,
            "participants": TEST_SUBJECTS,
            "frames_per_participant": TEST_FRAMES,
            "windows_per_fold_model": int(data["X"].shape[0]),
            "folds": N_FOLDS,
            "target_index": TARGET_INDEX,
            "stride": STRIDE,
            "param_limit": PARAM_LIMIT,
            "storage_limit_bytes": STORAGE_LIMIT_BYTES,
            "archive_limit_bytes": ARCHIVE_LIMIT_BYTES,
            "batch_probe_windows": PROBE_WINDOWS,
            "batch_probe_tolerance": PROBE_TOL,
            "metrics_clipped_at_floor": True,
            "device": str(device),
            "skipped": skipped,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores["experiments"], indent=2))


if __name__ == "__main__":
    main()

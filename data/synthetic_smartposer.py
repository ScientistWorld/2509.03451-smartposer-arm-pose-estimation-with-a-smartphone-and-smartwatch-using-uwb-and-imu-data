"""Synthetic SmartPoser-style arm tracking data.

The original SmartPoser dataset/code release is currently unavailable at the
paper URL. This generator is a data substitute only: it keeps the same problem
structure (phone/watch orientations and accelerations, raw UWB range with
structured body-occlusion error, corrected range target, and 3-D arm joints).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FPS = 25
WINDOW = 125


def _rot_x(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3), dtype=np.float32)
    out[..., 0, 0] = 1
    out[..., 1, 1] = c
    out[..., 1, 2] = -s
    out[..., 2, 1] = s
    out[..., 2, 2] = c
    return out


def _rot_y(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3), dtype=np.float32)
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3), dtype=np.float32)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1
    return out


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...jk->...ik", a, b).astype(np.float32)


def _smooth_angles(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    knots = max(8, n // 45)
    values = rng.uniform(low, high, size=knots)
    x = np.linspace(0, knots - 1, n)
    base = np.interp(x, np.arange(knots), values)
    wiggle = 0.12 * np.sin(np.linspace(0, rng.uniform(8, 16) * np.pi, n) + rng.uniform(0, np.pi))
    return (base + wiggle).astype(np.float32)


def _windowize(arrays: dict[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    xs = {k: [] for k in arrays}
    step = WINDOW if split == "train" else 25
    for start in range(0, arrays["features"].shape[0] - WINDOW + 1, step):
        for key, val in arrays.items():
            xs[key].append(val[start : start + WINDOW])
    return {k: np.stack(v).astype(np.float32) for k, v in xs.items()}


def _make_sequence(rng: np.random.Generator, frames: int, subject: int) -> dict[str, np.ndarray]:
    arm_span = rng.normal(1.70, 0.08)
    upper = arm_span * rng.normal(0.18, 0.012)
    fore = arm_span * rng.normal(0.165, 0.012)
    shoulder = np.zeros((frames, 3), dtype=np.float32)

    yaw = _smooth_angles(rng, frames, -1.35, 1.35)
    pitch = _smooth_angles(rng, frames, -1.05, 1.45)
    elbow_flex = _smooth_angles(rng, frames, 0.25, 2.45)
    wrist_roll = _smooth_angles(rng, frames, -1.2, 1.2)

    upper_rot = _matmul(_rot_z(yaw), _rot_y(pitch))
    fore_rot = _matmul(upper_rot, _rot_y(-elbow_flex))
    upper_vec = np.einsum("...ij,j->...i", upper_rot, np.array([upper, 0, 0], dtype=np.float32))
    fore_vec = np.einsum("...ij,j->...i", fore_rot, np.array([fore, 0, 0], dtype=np.float32))
    elbow = shoulder + upper_vec
    wrist = elbow + fore_vec
    joints = np.concatenate([shoulder, elbow, wrist], axis=1) / arm_span

    # Phone in left-front pocket in shoulder-centered coordinates.
    phone_pos = np.array([-0.10, -0.48, -0.17], dtype=np.float32) * (arm_span / 1.70)
    true_dist = np.linalg.norm(wrist - phone_pos[None, :], axis=1) / arm_span

    body_occ = ((wrist[:, 0] < -0.05) & (wrist[:, 2] < 0.03)) | ((wrist[:, 1] > 0.12) & (wrist[:, 0] < 0.10))
    structured = body_occ.astype(np.float32) * rng.uniform(0.10, 0.28)
    raw_dist = true_dist + structured + rng.normal(0, 0.025, size=frames).astype(np.float32)
    raw_dist = np.maximum(raw_dist, 0.03)

    phone_yaw = np.full(frames, rng.normal(0.0, 0.08), dtype=np.float32)
    phone_rot = _rot_z(phone_yaw)
    watch_rot = _matmul(fore_rot, _rot_x(wrist_roll))

    watch_vel = np.gradient(wrist, 1 / FPS, axis=0)
    watch_acc = np.gradient(watch_vel, 1 / FPS, axis=0) / 30.0
    phone_acc = rng.normal(0, 0.01, size=(frames, 3)).astype(np.float32)

    # Evaluation remains method-agnostic; methods receive only sensor features.
    features = np.concatenate(
        [
            raw_dist[:, None],
            phone_rot.reshape(frames, 9),
            watch_rot.reshape(frames, 9),
            phone_acc,
            watch_acc.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "features": features,
        "true_distance": true_dist[:, None].astype(np.float32),
        "joints": joints.astype(np.float32),
        "arm_span": np.full((frames, 1), arm_span, dtype=np.float32),
        "subject": np.full((frames, 1), subject, dtype=np.float32),
    }


def build_dataset(out_dir: Path, subjects: int = 10, frames: int = 1000, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_parts, test_parts = [], []
    for sid in range(subjects):
        seq = _make_sequence(rng, frames, sid)
        if sid == subjects - 1:
            test_parts.append(seq)
        else:
            train_parts.append(seq)
    train = {k: np.concatenate([p[k] for p in train_parts], axis=0) for k in train_parts[0]}
    test = {k: np.concatenate([p[k] for p in test_parts], axis=0) for k in test_parts[0]}
    np.savez_compressed(out_dir / "train_windows.npz", **_windowize(train, "train"))
    np.savez_compressed(out_dir / "test_windows.npz", **_windowize(test, "test"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/smartposer_synthetic"))
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build_dataset(args.out, args.subjects, args.frames, args.seed)


if __name__ == "__main__":
    main()

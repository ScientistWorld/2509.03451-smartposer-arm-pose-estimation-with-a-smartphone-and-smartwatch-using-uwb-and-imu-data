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
    # Paper trains on non-overlapping 5 s windows, but at test time uses a
    # rolling buffer with one new frame per prediction.
    step = WINDOW if split == "train" else 1
    for start in range(0, arrays["features"].shape[0] - WINDOW + 1, step):
        for key, val in arrays.items():
            xs[key].append(val[start : start + WINDOW])
    return {k: np.stack(v).astype(np.float32) for k, v in xs.items()}


def _angle_ranges(profile: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
    if profile == "daily":
        # Front-of-body motions approximate the paper's daily activity phase.
        return (-0.90, 1.05), (-0.70, 0.95), (0.35, 2.05), 0.80
    if profile == "gross":
        # Wider, more extreme poses approximate the gross motor phase.
        return (-1.55, 1.55), (-1.25, 1.65), (0.15, 2.65), 1.20
    if profile == "combined":
        return (-1.35, 1.35), (-1.05, 1.45), (0.25, 2.45), 1.00
    raise ValueError(f"unknown profile {profile!r}; expected daily, gross, or combined")


def _make_sequence(rng: np.random.Generator, frames: int, subject: int, profile: str) -> dict[str, np.ndarray]:
    arm_span = rng.normal(1.70, 0.08)
    upper = arm_span * rng.normal(0.18, 0.012)
    fore = arm_span * rng.normal(0.165, 0.012)
    shoulder = np.zeros((frames, 3), dtype=np.float32)

    yaw_range, pitch_range, elbow_range, uwb_bias_scale = _angle_ranges(profile)
    yaw = _smooth_angles(rng, frames, *yaw_range)
    pitch = _smooth_angles(rng, frames, *pitch_range)
    elbow_flex = _smooth_angles(rng, frames, *elbow_range)
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

    # Structured UWB range error: body occlusion and multipath produce a
    # pose-dependent positive bias, not just white noise. This is the phenomenon
    # SmartPoser's first RNN is meant to correct.
    behind_torso = wrist[:, 0] < 0.05
    near_head = (wrist[:, 1] > 0.08) & (wrist[:, 2] < 0.20)
    across_body = (wrist[:, 0] < 0.22) & (wrist[:, 1] > -0.05)
    body_occ = behind_torso | near_head | across_body
    occ_strength = (
        0.07
        + 0.11 / (1.0 + np.exp(18.0 * (wrist[:, 0] - 0.12)))
        + 0.05 * np.maximum(wrist[:, 1] + 0.05, 0.0)
    ).astype(np.float32)
    burst = 0.025 * (np.sin(np.linspace(0, rng.uniform(4, 8) * np.pi, frames) + rng.uniform(0, np.pi)) + 1.0)
    structured = body_occ.astype(np.float32) * (occ_strength + burst.astype(np.float32)) * uwb_bias_scale
    raw_dist_continuous = true_dist + structured + rng.normal(0, 0.035, size=frames).astype(np.float32)
    raw_dist_continuous = np.maximum(raw_dist_continuous, 0.03)
    # UWB arrives at roughly 5 Hz in the paper and is upsampled to the 25 Hz
    # IMU stream by repeating the latest range measurement.
    uwb_stride = FPS // 5
    sample_idx = (np.arange(frames) // uwb_stride) * uwb_stride
    raw_dist = raw_dist_continuous[sample_idx]

    phone_yaw = np.full(frames, rng.normal(0.0, 0.08), dtype=np.float32)
    phone_rot = _rot_z(phone_yaw)
    watch_rot = _matmul(fore_rot, _rot_x(wrist_roll))

    watch_vel = np.gradient(wrist, 1 / FPS, axis=0)
    watch_acc = np.gradient(watch_vel, 1 / FPS, axis=0) / 30.0
    phone_acc = rng.normal(0, 0.01, size=(frames, 3)).astype(np.float32)
    phone_rot_t = np.swapaxes(phone_rot, -1, -2)
    watch_rot_rel = _matmul(phone_rot_t, watch_rot)
    watch_acc_rel = np.einsum("...ij,...j->...i", phone_rot_t, watch_acc).astype(np.float32)

    # Evaluation remains method-agnostic; methods receive only sensor features.
    features = np.concatenate(
        [
            raw_dist[:, None],
            phone_rot.reshape(frames, 9),
            watch_rot_rel.reshape(frames, 9),
            phone_acc,
            watch_acc_rel,
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


def build_dataset(
    out_dir: Path,
    subjects: int = 10,
    frames: int = 1000,
    seed: int = 7,
    test_subject: int | None = None,
    profile: str = "combined",
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    if test_subject is None:
        test_subject = subjects - 1
    if not 0 <= test_subject < subjects:
        raise ValueError(f"test_subject must be in [0, {subjects}), got {test_subject}")
    train_parts, test_parts = [], []
    for sid in range(subjects):
        seq = _make_sequence(rng, frames, sid, profile)
        if sid == test_subject:
            test_parts.append(seq)
        else:
            train_parts.append(seq)
    train = {k: np.concatenate([p[k] for p in train_parts], axis=0) for k in train_parts[0]}
    test = {k: np.concatenate([p[k] for p in test_parts], axis=0) for k in test_parts[0]}
    np.savez_compressed(out_dir / "train_windows.npz", **_windowize(train, "train"))
    np.savez_compressed(out_dir / "test_windows.npz", **_windowize(test, "test"))
    (out_dir / "metadata.json").write_text(
        f'{{"subjects": {subjects}, "frames": {frames}, "seed": {seed}, "test_subject": {test_subject}, "profile": "{profile}"}}\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/smartposer_synthetic"))
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-subject", type=int, default=None)
    ap.add_argument("--profile", choices=["combined", "daily", "gross"], default="combined")
    args = ap.parse_args()
    build_dataset(args.out, args.subjects, args.frames, args.seed, args.test_subject, args.profile)


if __name__ == "__main__":
    main()

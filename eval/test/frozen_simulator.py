"""Phone/watch arm-tracking simulator (substitute for the unavailable SmartPoser
UWB/IMU/Kinect release).

This file is private to the hidden test. It generates the hidden test
participants at evaluation time (with a private seed held in ``evaluate.py``)
and it is the only code that can produce data from this distribution. The
visible development participants were written once with ``--write-dev`` and are
committed under ``data/smartposer_dev``.

Each simulated participant performs a "daily activities" phase followed by a
"gross motor" phase (wider, more occlusion-prone arm motion), mirroring the two
phases of the paper's study. Streams are at 25 Hz:

* UWB phone-watch range, sampled at 5 Hz and held to 25 Hz, with
  pose-dependent body-occlusion bias and white noise.
* Phone and watch orientation estimates with slowly drifting heading error and
  small tilt error (inertial drift), watch orientation expressed in the phone
  frame.
* Accelerations with white noise and slowly varying bias, rotated with the
  drifting orientation estimates, expressed in the phone frame and divided by 30.

Targets are shoulder-frame shoulder/elbow/wrist positions divided by arm span,
and the true phone-watch distance divided by arm span.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SIMULATOR_VERSION = "smartposer-sim-v2"
FPS = 25
UWB_HZ = 5
FEATURE_DIM = 25
# (name, yaw range, pitch range, elbow-flex range, UWB occlusion-bias scale)
PHASES = (
    ("daily", (-0.90, 1.05), (-0.70, 0.95), (0.35, 2.05), 0.80),
    ("gross", (-1.55, 1.55), (-1.25, 1.65), (0.15, 2.65), 1.20),
)
DEV_SEED = 7
DEV_SUBJECTS = 10
DEV_FRAMES = 12000


def _rot_x(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3))
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = c
    out[..., 1, 2] = -s
    out[..., 2, 1] = s
    out[..., 2, 2] = c
    return out


def _rot_y(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3))
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(a.shape + (3, 3))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...jk->...ik", a, b)


def _apply(rot: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", rot, vec)


def _euler_zyx(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> np.ndarray:
    return _matmul(_matmul(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma))
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _knot_track(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    times = [0.0]
    while times[-1] < n - 1:
        times.append(times[-1] + rng.uniform(1.2, 2.4) * FPS)
    values = rng.uniform(low, high, size=len(times))
    return np.interp(np.arange(n, dtype=np.float64), np.asarray(times), values)


def _angle_track(rng: np.random.Generator, lengths: tuple[int, int], ranges: list[tuple[float, float]]) -> np.ndarray:
    parts = [_knot_track(rng, n, lo, hi) for n, (lo, hi) in zip(lengths, ranges)]
    track = _smooth(np.concatenate(parts), sigma=0.3 * FPS)
    t = np.arange(track.size, dtype=np.float64) / FPS
    wiggle = 0.10 * np.sin(2.0 * np.pi * rng.uniform(0.25, 0.70) * t + rng.uniform(0.0, 2.0 * np.pi))
    return track + wiggle


def _ou(rng: np.random.Generator, n: int, std: float, tau_s: float, dims: int = 1) -> np.ndarray:
    """Stationary Ornstein-Uhlenbeck process (slow drift) with the given std."""
    a = float(np.exp(-1.0 / (tau_s * FPS)))
    innovations = rng.normal(0.0, std * np.sqrt(1.0 - a * a), size=(n, dims))
    out = np.empty((n, dims))
    out[0] = rng.normal(0.0, std, size=dims)
    for i in range(1, n):
        out[i] = a * out[i - 1] + innovations[i]
    return out


def make_subject(rng: np.random.Generator, frames: int, subject: int) -> dict[str, np.ndarray]:
    n_daily = frames // 2
    lengths = (n_daily, frames - n_daily)
    phase = np.concatenate([np.zeros(lengths[0], dtype=np.int8), np.ones(lengths[1], dtype=np.int8)])

    arm_span = float(np.clip(rng.normal(1.70, 0.08), 1.50, 1.95))
    upper = arm_span * rng.normal(0.18, 0.012)
    fore = arm_span * rng.normal(0.165, 0.012)

    yaw = _angle_track(rng, lengths, [p[1] for p in PHASES])
    pitch = _angle_track(rng, lengths, [p[2] for p in PHASES])
    elbow_flex = np.clip(_angle_track(rng, lengths, [p[3] for p in PHASES]), 0.05, 2.75)
    wrist_roll = _angle_track(rng, lengths, [(-1.2, 1.2), (-1.2, 1.2)])
    bias_scale = np.array([p[4] for p in PHASES])[phase]

    upper_rot = _matmul(_rot_z(yaw), _rot_y(pitch))
    fore_rot = _matmul(upper_rot, _rot_y(-elbow_flex))
    elbow = upper * upper_rot[:, :, 0]
    wrist = elbow + fore * fore_rot[:, :, 0]
    shoulder = np.zeros_like(elbow)
    joints = np.concatenate([shoulder, elbow, wrist], axis=1) / arm_span

    # UWB range: phone in the left-front pocket, shoulder-centred coordinates.
    phone_pos = np.array([-0.10, -0.48, -0.17]) * (arm_span / 1.70)
    true_dist = np.linalg.norm(wrist - phone_pos[None, :], axis=1) / arm_span
    behind_torso = wrist[:, 0] < 0.05
    near_head = (wrist[:, 1] > 0.08) & (wrist[:, 2] < 0.20)
    across_body = (wrist[:, 0] < 0.22) & (wrist[:, 1] > -0.05)
    occluded = behind_torso | near_head | across_body
    occ_strength = 0.07 + 0.11 / (1.0 + np.exp(18.0 * (wrist[:, 0] - 0.12))) + 0.05 * np.maximum(wrist[:, 1] + 0.05, 0.0)
    t = np.arange(frames, dtype=np.float64) / FPS
    burst = 0.025 * (np.sin(2.0 * np.pi * rng.uniform(0.05, 0.10) * t + rng.uniform(0.0, 2.0 * np.pi)) + 1.0)
    structured = occluded.astype(np.float64) * (occ_strength + burst) * bias_scale
    raw_continuous = np.maximum(true_dist + structured + rng.normal(0.0, 0.035, size=frames), 0.03)
    stride = FPS // UWB_HZ
    raw_dist = raw_continuous[(np.arange(frames) // stride) * stride]

    # Orientation estimates with inertial drift.
    phone_true = _rot_z(np.full(frames, rng.normal(0.0, 0.08)))
    watch_true = _matmul(fore_rot, _rot_x(wrist_roll))
    w_heading = _ou(rng, frames, 0.12, 30.0)[:, 0]
    w_tilt = _ou(rng, frames, 0.03, 10.0, 2)
    watch_err = _euler_zyx(w_heading, w_tilt[:, 0], w_tilt[:, 1])
    p_heading = _ou(rng, frames, 0.06, 60.0)[:, 0]
    p_tilt = _ou(rng, frames, 0.015, 20.0, 2)
    phone_err = _euler_zyx(p_heading, p_tilt[:, 0], p_tilt[:, 1])
    watch_est = _matmul(watch_err, watch_true)
    phone_est = _matmul(phone_err, phone_true)
    phone_est_t = np.swapaxes(phone_est, -1, -2)
    watch_rel = _matmul(phone_est_t, watch_est)

    # Accelerations (m/s^2): device-frame sensor error rotated by the estimate.
    dt = 1.0 / FPS
    acc_true = np.gradient(np.gradient(wrist, dt, axis=0), dt, axis=0)
    watch_sensor_err = rng.normal(0.0, 0.35, size=(frames, 3)) + _ou(rng, frames, 0.12, 20.0, 3)
    acc_world_est = _apply(watch_err, acc_true) + _apply(watch_est, watch_sensor_err)
    watch_acc_rel = _apply(phone_est_t, acc_world_est) / 30.0
    phone_acc = (rng.normal(0.0, 0.15, size=(frames, 3)) + _ou(rng, frames, 0.05, 30.0, 3)) / 30.0

    features = np.concatenate(
        [
            raw_dist[:, None],
            phone_est.reshape(frames, 9),
            watch_rel.reshape(frames, 9),
            phone_acc,
            watch_acc_rel,
        ],
        axis=1,
    ).astype(np.float32)
    assert features.shape == (frames, FEATURE_DIM)
    return {
        "features": features,
        "joints": joints.astype(np.float32),
        "true_distance": true_dist[:, None].astype(np.float32),
        "arm_span": np.asarray(arm_span, dtype=np.float32),
        "phase": phase,
    }


def make_participants(seed: int, subjects: int, frames: int) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [make_subject(rng, frames, sid) for sid in range(subjects)]


def write_dev(out_dir: Path, subjects: int = DEV_SUBJECTS, frames: int = DEV_FRAMES, seed: int = DEV_SEED) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sums = []
    for sid, subject in enumerate(make_participants(seed, subjects, frames)):
        path = out_dir / f"subject_{sid:02d}.npz"
        np.savez_compressed(path, **subject)
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    meta = {
        "simulator": SIMULATOR_VERSION,
        "subjects": subjects,
        "frames_per_subject": frames,
        "fps": FPS,
        "feature_layout": [
            "0: raw UWB phone-watch range / arm span (5 Hz, held)",
            "1-9: phone orientation estimate (row-major 3x3)",
            "10-18: watch orientation estimate in the phone frame (row-major 3x3)",
            "19-21: phone acceleration / 30",
            "22-24: watch acceleration in the phone frame / 30",
        ],
        "targets": {
            "joints": "shoulder, elbow, wrist xyz in the shoulder frame / arm span (9 values)",
            "true_distance": "true phone-watch distance / arm span",
            "arm_span": "participant arm span in metres",
            "phase": "0 = daily-activity phase, 1 = gross-motor phase",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-dev", type=Path, required=True, help="Directory for the visible development participants.")
    args = ap.parse_args()
    write_dev(args.write_dev)


if __name__ == "__main__":
    main()

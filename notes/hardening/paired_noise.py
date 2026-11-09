"""Paired test-noise analysis on the hidden slice (iteration 11 hardening).

Every candidate's 10 fold models are run on the hidden windows with the hidden
evaluator's own builder and runner (eval/test/evaluate.py); per-window errors
are kept. Participants are then bootstrapped (cluster bootstrap, the same
resample for every candidate), and each candidate's rank difference to the
reference is recomputed per replicate with the coefficients and weights in
scoring/reference.json. The spread of that paired difference is the test's
noise for a comparison in which every method sees the same inputs.

Ranks are reported raw (weight x |coefficient| x improvement) and normalized
(each metric's improvement divided by its standard deviation across the valid
candidates on the full test, standing in for the spread across future attempts).

Usage:
  python notes/hardening/paired_noise.py --variants tmp/h2/variants --reference smartposer \
      --out notes/hardening/paired_noise.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "test"))
import evaluate as ev  # noqa: E402  (the hidden evaluator itself)

POSE = "pose_tracking_synthetic"
RANGE = "uwb_correction_synthetic"


def per_window(paths: list[Path], data: dict, device: torch.device, order: np.ndarray) -> dict[str, np.ndarray]:
    span = data["span"]
    elbow, wrist, rng = [], [], []
    for path in paths:
        r, j = ev.run_model(path, data["X"], device, order)
        err = np.linalg.norm((j - data["joints"]).reshape(-1, 3, 3), axis=-1) * span[:, None] * 100.0
        elbow.append(err[:, 1])
        wrist.append(err[:, 2])
        rng.append(np.abs(r - data["dist"]) * span * 100.0)
    return {"elbow": np.stack(elbow), "wrist": np.stack(wrist), "range": np.stack(rng)}


def metrics(e: dict[str, np.ndarray], idx: np.ndarray, data: dict, raw_err: np.ndarray, with_range: bool) -> dict[str, float]:
    el = e["elbow"][:, idx]
    wr = e["wrist"][:, idx]
    both = np.concatenate([el.ravel(), wr.ravel()])
    hm = data["high_motion"][idx]
    per_frame = 0.5 * (el + wr)
    out = {
        f"{POSE}/median_elbow_cm": float(np.median(el)),
        f"{POSE}/median_wrist_cm": float(np.median(wr)),
        f"{POSE}/median_elbow_wrist_cm": float(np.median(both)),
        f"{POSE}/mean_elbow_wrist_cm": float(np.mean(both)),
        f"{POSE}/p90_elbow_wrist_cm": float(np.percentile(both, 90)),
        f"{POSE}/high_motion_mean_elbow_wrist_cm": float(np.mean(per_frame[:, hm])),
    }
    if with_range:
        c = e["range"][:, idx]
        raw = raw_err[idx]
        hard = data["hard_range"][idx]
        mae = float(np.mean(c))
        out.update(
            {
                f"{RANGE}/uwb_error_reduction_pct": 100.0 * (1.0 - mae / float(np.mean(raw))),
                f"{RANGE}/corrected_uwb_mae_cm": mae,
                f"{RANGE}/corrected_uwb_p90_cm": float(np.percentile(c, 90)),
                f"{RANGE}/occluded_uwb_mae_cm": float(np.mean(c[:, hard])),
                f"{RANGE}/uwb_correction_win_rate": float(np.mean(c < raw[None, :])),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--reference", default="smartposer")
    ap.add_argument("--replicates", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    definition = json.loads((ROOT / "scoring" / "reference.json").read_text())
    weights = {
        f"{exp}/{m}": (ed["weight"] * abs(md["coefficient"]), md["higher_is_better"])
        for exp, ed in definition["experiments"].items()
        for m, md in ed["metrics"].items()
        if md["coefficient"] != 0
    }

    data = ev.build_test_set()
    order = np.random.default_rng(ev.ORDER_SEED).permutation(data["X"].shape[0])
    raw_err = np.abs(data["raw"] - data["dist"]) * data["span"] * 100.0
    full = np.arange(data["X"].shape[0])

    cands: dict[str, list[Path]] = {}
    for d in sorted(p for p in args.variants.iterdir() if p.is_dir()):
        paths = [d / f"fold_{k}" / "model.pt" for k in range(ev.N_FOLDS)]
        if all(p.exists() for p in paths):
            cands[d.name] = paths
    fixed = [ev.FIXED_DIR / "imu_only" / f"fold_{k}.pt" for k in range(ev.N_FOLDS)]

    errors = {name: per_window(paths, data, device, order) for name, paths in cands.items()}
    imu = per_window(fixed, data, device, order) if all(p.exists() for p in fixed) else None
    full_metrics = {name: metrics(e, full, data, raw_err, True) for name, e in errors.items()}

    # Validity on the full test, mirroring the evaluator's gates.
    floor_med = ev.pose_metrics(
        np.broadcast_to(data["joints"].mean(axis=0), data["joints"].shape), data["joints"], data["span"], data["high_motion"]
    )["median_elbow_wrist_cm"]
    floor_rng = float(np.mean(raw_err))
    valid = [
        n
        for n, m in full_metrics.items()
        if m[f"{POSE}/median_elbow_wrist_cm"] < floor_med and m[f"{RANGE}/corrected_uwb_mae_cm"] < floor_rng
    ]
    spreads = {k: statistics.pstdev([full_metrics[n][k] for n in valid]) for k in weights}

    def rank(m: dict, ref: dict) -> tuple[float, float]:
        raw = norm = 0.0
        for k, (w, hib) in weights.items():
            imp = (m[k] - ref[k]) if hib else (ref[k] - m[k])
            raw += w * imp
            if spreads[k] > 0:
                norm += w * imp / spreads[k]
        return raw, norm

    ref_full = full_metrics[args.reference]
    pid = data["participant"]
    groups = [np.flatnonzero(pid == p) for p in range(int(pid.max()) + 1)]
    rng = np.random.default_rng(2026)
    boot: dict[str, list[tuple[float, float]]] = {n: [] for n in valid if n != args.reference}
    paper_effect: list[float] = []
    for _ in range(args.replicates):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[p] for p in pick])
        ref_m = metrics(errors[args.reference], idx, data, raw_err, True)
        for n in boot:
            boot[n].append(rank(metrics(errors[n], idx, data, raw_err, True), ref_m))
        if imu is not None:
            imu_m = metrics(imu, idx, data, raw_err, False)
            paper_effect.append(imu_m[f"{POSE}/median_elbow_wrist_cm"] - ref_m[f"{POSE}/median_elbow_wrist_cm"])

    report: dict = {
        "reference": args.reference,
        "participants": len(groups),
        "windows_per_fold_model": int(full.size),
        "replicates": args.replicates,
        "valid": valid,
        "spreads": spreads,
        "candidates": {},
    }
    for n in sorted(valid, key=lambda n: -rank(full_metrics[n], ref_full)[1]):
        raw_obs, norm_obs = rank(full_metrics[n], ref_full)
        row = {
            "median_elbow_wrist_cm": full_metrics[n][f"{POSE}/median_elbow_wrist_cm"],
            "uwb_error_reduction_pct": full_metrics[n][f"{RANGE}/uwb_error_reduction_pct"],
            "rank_raw": raw_obs,
            "rank_normalized": norm_obs,
        }
        if n in boot:
            b = np.asarray(boot[n])
            row.update(
                {
                    "paired_sd_raw": float(b[:, 0].std(ddof=1)),
                    "paired_sd_normalized": float(b[:, 1].std(ddof=1)),
                    "ci95_normalized": [float(np.percentile(b[:, 1], 2.5)), float(np.percentile(b[:, 1], 97.5))],
                    "p_better_than_reference": float(np.mean(b[:, 1] > 0)),
                }
            )
        report["candidates"][n] = row
    if paper_effect:
        pe = np.asarray(paper_effect)
        imu_full = metrics(imu, full, data, raw_err, False)[f"{POSE}/median_elbow_wrist_cm"]
        report["paper_effect_imu_minus_reference_median_cm"] = {
            "observed": imu_full - ref_full[f"{POSE}/median_elbow_wrist_cm"],
            "paired_sd": float(pe.std(ddof=1)),
            "ci95": [float(np.percentile(pe, 2.5)), float(np.percentile(pe, 97.5))],
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"Paired participant bootstrap on the hidden slice ({len(groups)} participants, {args.replicates} replicates); reference = `{args.reference}`.",
        "",
        "| candidate | median e/w cm | range error removed % | rank raw | rank normalized | paired SD (norm.) | 95% CI (norm.) | P(better) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in report["candidates"].items():
        ci = r.get("ci95_normalized")
        lines.append(
            f"| {n} | {r['median_elbow_wrist_cm']:.3f} | {r['uwb_error_reduction_pct']:.2f} | {r['rank_raw']:+.3f} | {r['rank_normalized']:+.3f} | "
            + (f"{r['paired_sd_normalized']:.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {r['p_better_than_reference']:.2f} |" if ci else "— | — | — |")
        )
    if paper_effect:
        pe = report["paper_effect_imu_minus_reference_median_cm"]
        lines += ["", f"Paper effect (IMU-only minus reference median e/w): {pe['observed']:.3f} cm, paired SD {pe['paired_sd']:.3f}, 95% CI [{pe['ci95'][0]:.3f}, {pe['ci95'][1]:.3f}]."]
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Pick the training length of the reference and the IMU-only baseline from the
visible (dev, leave-one-participant-out) snapshot scores only.

Usage: python notes/hardening/select_epoch.py notes/hardening/snapshots_scores_train.json
"""

from __future__ import annotations

import json
import sys


def curve(status: dict, prefix: str, require_valid: bool) -> dict[int, float]:
    out = {}
    for name, st in status.items():
        if not name.startswith(prefix + "_e") or "pose" not in st:
            continue
        if require_valid and not st.get("valid", False):
            continue
        out[int(name.rsplit("_e", 1)[1])] = st["pose"]["median_elbow_wrist_cm"]
    return dict(sorted(out.items()))


def main() -> None:
    scores = json.load(open(sys.argv[1]))
    status = scores["_status"]
    method_curve = curve(status, "smartposer", require_valid=True)
    baseline_curve = curve(status, "imu_only", require_valid=False)
    method_eligible = {e: v for e, v in method_curve.items() if e >= 25}
    baseline_eligible = {e: v for e, v in baseline_curve.items() if e >= 25}
    result = {
        "rule": "epoch with the lowest dev leave-one-participant-out median elbow/wrist error (snapshots >= 25 epochs)",
        "smartposer_epochs": min(method_eligible, key=method_eligible.get),
        "imu_only_epochs": min(baseline_eligible, key=baseline_eligible.get),
        "smartposer_dev_curve_cm": method_curve,
        "imu_only_dev_curve_cm": baseline_curve,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

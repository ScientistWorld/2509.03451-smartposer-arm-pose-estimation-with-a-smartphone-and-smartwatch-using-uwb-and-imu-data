"""Rank candidate rows against the stored reference with the gym's rank rule.

rank = sum over experiments e and scored metrics m of
       weight_e * |coefficient_m| * improvement_m [/ spread_m]
where improvement_m = candidate - reference in the metric's better direction
(``higher_is_better``) and the reference is the ``smartposer`` row of the
stored scoring/scores_{test,train}.json. ``spread_m`` (normalized rank only) is
the standard deviation of metric m across the valid attempts listed in the
variants file (reference re-runs, longer/shorter training, architecture and
training variants), standing in for the spread across future attempts.

Usage: python notes/hardening/rank.py [OUT_DIR]
  (reads <OUT_DIR>/{variants,shortcuts,snapshots}_scores_{test,train}.json,
   default notes/hardening; writes <OUT_DIR>/ranks.json; prints markdown)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
H = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "notes" / "hardening"
REF_METHOD = "smartposer"
FIXED = {"mean_pose", "raw_uwb", "imu_only"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def candidate_rows(scores: dict) -> dict[str, dict[str, dict]]:
    rows: dict[str, dict[str, dict]] = {}
    for exp, data in scores["experiments"].items():
        for method, metrics in data["results"].items():
            if method in FIXED:
                continue
            rows.setdefault(method, {})[exp] = metrics
    return rows


def improvements(definition: dict, ref_row: dict, cand_row: dict) -> dict[str, float]:
    out = {}
    for exp, ed in definition["experiments"].items():
        for m, md in ed["metrics"].items():
            if md["coefficient"] == 0:
                continue
            delta = cand_row[exp][m] - ref_row[exp][m]
            out[f"{exp}/{m}"] = delta if md["higher_is_better"] else -delta
    return out


def weights(definition: dict) -> dict[str, float]:
    return {
        f"{exp}/{m}": ed["weight"] * abs(md["coefficient"])
        for exp, ed in definition["experiments"].items()
        for m, md in ed["metrics"].items()
        if md["coefficient"] != 0
    }


def rank_table(definition: dict, ref_scores: dict, scores: dict, spreads: dict[str, float] | None) -> dict[str, dict]:
    ref_rows = candidate_rows(ref_scores)
    ref_row = ref_rows[REF_METHOD]
    w = weights(definition)
    table = {}
    for method, row in candidate_rows(scores).items():
        imp = improvements(definition, ref_row, row)
        raw = sum(w[k] * v for k, v in imp.items())
        norm = None
        if spreads:
            norm = sum(w[k] * v / spreads[k] for k, v in imp.items() if spreads.get(k, 0) > 0)
        status = scores.get("_status", {}).get(method, {})
        table[method] = {
            "valid": status.get("valid"),
            "reasons": status.get("reasons", []),
            "param_count": status.get("param_count"),
            "median_elbow_wrist_cm": row["pose_tracking_synthetic"]["median_elbow_wrist_cm"],
            "corrected_uwb_mae_cm": row["uwb_correction_synthetic"]["corrected_uwb_mae_cm"],
            "measured_median_elbow_wrist_cm": status.get("pose", {}).get("median_elbow_wrist_cm"),
            "measured_corrected_uwb_mae_cm": status.get("range", {}).get("corrected_uwb_mae_cm"),
            "rank_raw": raw,
            "rank_normalized": norm,
        }
    return table


def spreads_from(definition: dict, scores: dict) -> dict[str, float]:
    rows = candidate_rows(scores)
    valid = [m for m in rows if scores.get("_status", {}).get(m, {}).get("valid")]
    out = {}
    for exp, ed in definition["experiments"].items():
        for m, md in ed["metrics"].items():
            if md["coefficient"] == 0:
                continue
            vals = [rows[name][exp][m] for name in valid]
            out[f"{exp}/{m}"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return out


def fmt(x, nd=3) -> str:
    return "—" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def main() -> None:
    definition = load(ROOT / "scoring" / "reference.json")
    report: dict = {}
    for slice_name in ("test", "train"):
        ref_scores = load(ROOT / "scoring" / f"scores_{slice_name}.json")
        variants_path = H / f"variants_scores_{slice_name}.json"
        if not variants_path.exists():
            continue
        variants = load(variants_path)
        spreads = spreads_from(definition, variants)
        report[slice_name] = {"spreads": spreads, "variants": rank_table(definition, ref_scores, variants, spreads)}
        for extra in ("shortcuts", "snapshots", "protocol"):
            path = H / f"{extra}_scores_{slice_name}.json"
            if path.exists():
                report[slice_name][extra] = rank_table(definition, ref_scores, load(path), spreads)
        seeds = [report[slice_name]["variants"][m] for m in ("smartposer", "smartposer_seed18", "smartposer_seed19") if m in report[slice_name]["variants"]]
        if seeds:
            report[slice_name]["seed_rank_raw_range"] = max(s["rank_raw"] for s in seeds) - min(s["rank_raw"] for s in seeds)
            report[slice_name]["seed_rank_normalized_range"] = max(s["rank_normalized"] for s in seeds) - min(s["rank_normalized"] for s in seeds)
        if "smartposer_rerun17" in report[slice_name]["variants"]:
            r = report[slice_name]["variants"]["smartposer_rerun17"]
            report[slice_name]["same_seed_rerun_rank"] = {"raw": r["rank_raw"], "normalized": r["rank_normalized"]}
    # Fixed-baseline integrity: the imu_only row in shortcut runs must equal the stored one.
    sc_path = H / "shortcuts_scores_test.json"
    if sc_path.exists():
        stored = load(ROOT / "scoring" / "scores_test.json")["experiments"]["pose_tracking_synthetic"]["results"].get("imu_only")
        seen = load(sc_path)["experiments"]["pose_tracking_synthetic"]["results"].get("imu_only")
        report["fixed_baseline_unchanged_under_replacement"] = stored == seen
    (H / "ranks.json").write_text(json.dumps(report, indent=2) + "\n")

    for slice_name, data in report.items():
        if not isinstance(data, dict) or "variants" not in data:
            continue
        print(f"\n## {slice_name} slice (reference = stored `{REF_METHOD}` row)\n")
        if "seed_rank_raw_range" in data:
            print(f"Seed range of rank: raw {data['seed_rank_raw_range']:.3f}, normalized {data['seed_rank_normalized_range']:.3f}\n")
        if "same_seed_rerun_rank" in data:
            rr = data["same_seed_rerun_rank"]
            print(f"Same-seed rerun of the reference: raw {rr['raw']:+.3f}, normalized {rr['normalized']:+.3f}\n")
        for group in ("variants", "snapshots", "shortcuts", "protocol"):
            if group not in data:
                continue
            print(f"### {group}\n")
            print("| candidate | valid | params | median e/w cm | range MAE cm | rank (raw) | rank (normalized) |")
            print("|---|---|---:|---:|---:|---:|---:|")
            for name, r in sorted(data[group].items(), key=lambda kv: -kv[1]["rank_raw"]):
                print(
                    f"| {name} | {r['valid']} | {r['param_count']} | {r['median_elbow_wrist_cm']:.3f} | "
                    f"{r['corrected_uwb_mae_cm']:.3f} | {fmt(r['rank_raw'])} | {fmt(r['rank_normalized'])} |"
                )
            print()


if __name__ == "__main__":
    main()

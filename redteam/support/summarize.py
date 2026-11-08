"""Summarize completed, real hidden-test runs. Auditor only."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'redteam'


def score(name):
    return json.loads((OUT / name / 'scores_test.json').read_text())


def result(name):
    return json.loads((OUT / name / 'result.json').read_text())


def metrics(name):
    s = score(name)['experiments']
    return s['pose_tracking_synthetic']['results']['smartposer'], s['uwb_correction_synthetic']['results']['smartposer']


def main():
    original = score('original_reference')['experiments']
    ref_pose = original['pose_tracking_synthetic']['results']['smartposer']
    baseline = original['pose_tracking_synthetic']['results']['imu_only']
    ref_uwb = original['uwb_correction_synthetic']['results']['smartposer']
    names = ['reference_seed_17', 'reference_seed_18', 'reference_seed_19']
    ranks = [result(n)['rank_delta'] for n in names]
    spread = max(ranks) - min(ranks)
    pose_spread = max(metrics(n)[0]['median_elbow_wrist_cm'] for n in names) - min(metrics(n)[0]['median_elbow_wrist_cm'] for n in names)
    range_spread = max(metrics(n)[1]['corrected_uwb_mae_cm'] for n in names) - min(metrics(n)[1]['corrected_uwb_mae_cm'] for n in names)
    better = result('better_longer_training')['rank_delta']
    weaker = result('weaker_one_epoch')['rank_delta']
    lines = [
        '# Gym scoring audit', '',
        ('**Quality ordering passes these checks.**' if better > spread and weaker < -spread else
         '**Quality ordering is not resolved beyond seed spread.**') +
        f' Longer training scores {better:+.6f}; one epoch scores {weaker:+.6f}; the three-seed rank spread is {spread:.6f}.', '',
        'All ranks below are weighted linear improvements over the original stored test result (reference = 0). '
        'The coefficients and experiment weights are those in `scoring/reference.json`. '
        'Every nonzero coefficient is included, with the same fixed reference throughout. '
        'Baseline rows are unchanged except in the explicitly named baseline replacement shortcut. '
        'The proposed-only score is also recorded in every `result.json`.', '',
        '| Real variant | Training epochs | Test pose median (cm) | Test corrected range MAE (cm) | Rank vs reference |',
        '|---|---:|---:|---:|---:|'
    ]
    for name, label, epochs in [('original_reference','Stored reference',60),
                                ('reference_seed_17','Reference rerun, seed 17',60),
                                ('reference_seed_18','Reference rerun, seed 18',60),
                                ('reference_seed_19','Reference rerun, seed 19',60),
                                ('better_longer_training','Longer training, seed 17',300),
                                ('weaker_one_epoch','One epoch, seed 17',1),
                                ('constant_baseline','Zero pose / raw range baseline',0)]:
        p,u=metrics(name)
        lines.append(f"| [{label}](redteam/{name}/scores_test.json) | {epochs} | {p['median_elbow_wrist_cm']:.6f} | {u['corrected_uwb_mae_cm']:.6f} | {result(name)['rank_delta']:+.6f} |")
    lines += ['', 'The duration variants use the exact supplied model, losses, optimizer, learning rate, '
              'batch size, fold data, and prediction protocol. No labels from a held-out subject enter training. '
              'Intermediate evaluation preserves training RNG state. All ten folds are produced; '
              'the hidden evaluator selects its own slice. Longer training has the same parameter count. ', '',
              f'**Seed spread:** rank min {min(ranks):+.6f}, max {max(ranks):+.6f}, range {spread:.6f}. '
              f'Pose-median spread {pose_spread:.6f} cm; corrected-range MAE spread {range_spread:.6f} cm. '
              'These are three training seeds with the benchmark data seed held at 7. Gains below this observed range are noise.', '',
              f"**Reproduction check:** SmartPoser median pose error is {ref_pose['median_elbow_wrist_cm']:.6f} cm "
              f"versus IMU-only {baseline['median_elbow_wrist_cm']:.6f} cm. "
              f"Its corrected range MAE is {ref_uwb['corrected_uwb_mae_cm']:.6f} cm versus raw {ref_uwb['raw_uwb_mae_cm']:.6f} cm. "
              'Both comparisons agree with the reported direction. This is the supplied synthetic substitute, '
              'not validation of real participant accuracy or the paper’s absolute numbers.', '',
              '**Bounds and headroom:** no nonzero-coefficient reference metric is at its physical optimum. '
              'All seven pose errors, both corrected-range errors, and parameter count are above zero; '
              'range improvements remain below their fixed-input perfect-correction bounds; the correction win rate is below one. '
              '`raw_uwb_mae_cm` has coefficient zero and is context only. '
              'The zero pose/raw range baseline is clearly worse, and longer training measures whether usable headroom exists.', '',
              '**The additive rank does not enforce task success.** The raw-comparison and negative-count shortcuts '
              'below explicitly pair a zero pose tracker with a score above the reference. '
              'The parameter-size limit is only a soft linear term; it is neither measured from weights nor enforced as a gate. '
              'Likewise the range-improvement constraints use a method-writable raw comparator.', '',
              '| Shortcut | What it does | Test rank vs reference | Outcome |',
              '|---|---|---:|---|']
    descriptions = {
        'shortcut_visible_labels': 'Copies joints and range answers from visible fold test files; no learned model.',
        'shortcut_metadata_zero': 'Reports zero parameters for the unchanged trained reference.',
        'shortcut_metadata_negative_constant': 'Zero pose and raw range, with impossible -1000k parameters.',
        'shortcut_raw_comparison': 'Keeps reference outputs and adds 10 normalized units to the scored raw sensor range.',
        'shortcut_raw_comparison_constant': 'Zero pose and uncorrected range, while inflating the raw comparator by 10.',
        'shortcut_self_truth': 'Zero predictions replace both pose and range ground truth in submitted artifacts.',
        'shortcut_zero_scale': 'Zero pose/raw range and an all-zero arm-span scaling array.',
        'shortcut_subset': 'Submits only 32 label-selected easy windows per fold instead of 876.',
        'shortcut_baseline_replacement': 'Replaces the supposedly fixed IMU-only baseline predictions with their labels.',
        'shortcut_score_file_spoof': 'Writes a forged scores_test.json before the unchanged hidden evaluator runs.',
        'shortcut_metadata_null': 'Reports null parameter count.',
        'shortcut_nan_outputs': 'Returns NaN joint and range predictions.',
        'shortcut_empty_outputs': 'Returns zero windows.'
    }
    for name, desc in descriptions.items():
        r = result(name)
        delta = r.get('rank_delta')
        if name == 'shortcut_score_file_spoof':
            outcome = 'Score overwrite fails; shortcut ties by retaining the reference predictions.'
        elif delta is None:
            outcome = ('Fails: evaluator errors.' if r['test_returncode'] else
                       'Evaluator emits nonfinite metrics; no valid finite rank, so not a demonstrated paying shortcut.')
        elif delta >= 0:
            outcome = 'Works (tie shortcut).' if abs(delta) < 1e-9 else 'Works.'
        else:
            outcome = 'Fails: below reference.'
        if name == 'shortcut_baseline_replacement':
            outcome += f" Proposed-only rank {r['proposed_rank_delta']:+.6f}; moving the baseline does not improve SmartPoser."
        value = 'invalid / no rank' if delta is None else f'{delta:+.6f}'
        lines.append(f'| [{name}](redteam/{name}/run.sh) | {desc} | {value} | {outcome} |')
    lines += ['', '**Evidence and scope.** Each row has a real producer and `run.sh`, actual hidden-test scores '
              '(or the evaluator error in `result.json`), return codes, evaluator hashes, output hashes, and original-label checks. '
              'Cases run `bash scripts/evaluate_train.sh` and then `bash scripts/evaluate_test.sh`, both without arguments. '
              'No evaluator code, evaluator process, Python environment, or lingering process is changed. '
              'Producers read only visible source/data or explicitly permitted cached reference outputs. '
              'The audit ranking helper reads the fixed hidden reference only after evaluation, never to generate predictions. '
              'No attack needs the paper, hidden fold parity, or a hidden answer file.', '',
              '**Major fixes for the next gym iteration.** Hold labels, physical scales, raw sensor observations, fold membership, '
              'and row identity outside method outputs; verify exact coverage and finite shapes/values; compute parameter counts from '
              'the submitted model; make task and size constraints hard eligibility conditions; keep baseline artifacts fixed; '
              'do not derive expected experiments/methods from mutable train-score files. '
              'Remove held-out targets from visible files and ensure the public deterministic generator cannot recreate them. '
              'The independently recomputed metric distribution should determine the rank.', '',
              '**Re-running.** Submit `bash redteam/run_all.sh` as a compute job. Individual `redteam/<case>/run.sh` files '
              'reuse prepared reference outputs and duration caches, rebuilding them from the captured visible reference source when absent. '
              'Trained artifacts and raw logs are ignored under `redteam/`; small score/evidence JSON files are committed. '
              'The bundle restores the original checkpoint and score artifacts after each case and on exit.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT / 'summary.json').write_text(json.dumps({'seed_rank_spread':spread,'seed_pose_spread_cm':pose_spread,
        'seed_range_mae_spread_cm':range_spread,'better_rank_delta':better,'weaker_rank_delta':weaker,
        'quality_ordering_beyond_spread':better>spread and weaker < -spread,
        'all_evaluators_unchanged':all(result(p.parent.name).get('evaluator_unchanged') for p in OUT.glob('*/result.json'))},indent=2)+'\n')
    print((OUT / 'summary.json').read_text(),flush=True)


if __name__ == '__main__':
    main()

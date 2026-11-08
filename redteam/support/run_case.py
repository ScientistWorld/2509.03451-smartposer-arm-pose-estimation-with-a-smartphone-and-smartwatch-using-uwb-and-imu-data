"""Run artifact production and then the actual hidden test in a separate process.

The producer imports only visible method code and reads only visible data or
reusable reference predictions. The auditor's rank computation runs afterwards.
The evaluator and its Python environment are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from cases import ROOT, CACHE, install_original, install_variant, mutate_shortcut, load_arrays
from rank import rank
from prepare import prepare


def restore_scores(saved):
    for name, content in saved.items():
        p = ROOT / 'scoring' / name
        if content is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(content)


def ensure_variant(seed, epochs):
    expected = CACHE / f'seed_{seed}_epochs_{epochs}' / 'smartposer_fold_9/complete.json'
    if expected.exists():
        return
    subprocess.run([sys.executable, str(ROOT / 'redteam/support/train_variants.py'),
                    '--seed', str(seed), '--epochs', str(epochs)], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--epochs', type=int)
    args = ap.parse_args()
    dest = ROOT / 'redteam' / args.name
    dest.mkdir(exist_ok=True)
    saved = {name: (ROOT / 'scoring' / name).read_bytes() if (ROOT / 'scoring' / name).exists() else None
             for name in ['scores_train.json', 'scores_test.json']}
    protected = [ROOT / 'scripts/evaluate_test.sh', ROOT / 'scripts/evaluate_train.sh',
                 *sorted((ROOT / 'eval').rglob('*.py'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    status = {'case': args.name, 'test_command': ['bash', 'scripts/evaluate_test.sh'],
              'train_command': ['bash', 'scripts/evaluate_train.sh'], 'started': time.time(),
              'evaluator_hashes_before': hashes}
    try:
        prepare()
        if args.seed is not None:
            ensure_variant(args.seed, args.epochs)
        install_original()
        if args.seed is not None:
            install_variant(args.seed, args.epochs)
            status.update(seed=args.seed, epochs=args.epochs)
        elif args.name != 'original_reference':
            mutate_shortcut(args.name)
        # Record the method's output shape and its actual errors under ORIGINAL
        # labels/scales. This audit-only check is not the scored evaluator.
        counts, errors, ranges = [], [], []
        artifact_hashes = {}
        for fold in range(10):
            path = ROOT / 'checkpoints' / f'smartposer_fold_{fold}' / 'predictions.npz'
            artifact_hashes[path.parent.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            d = load_arrays(path)
            base = load_arrays(CACHE / 'original_checkpoints' / path.parent.name / 'predictions.npz')
            counts.append(int(d['pred_joints'].shape[0]))
            if d['pred_joints'].shape == base['true_joints'].shape:
                errors.append(np.linalg.norm((d['pred_joints'][:, 119].reshape(-1,3,3) -
                              base['true_joints'][:, 119].reshape(-1,3,3))[:,1:],axis=-1) *
                              base['arm_span'][:,119] * 100)
                ranges.append(np.abs(d['corrected_distance'] - base['true_distance']) *
                              base['arm_span'] * 100)
        status.update(windows_per_fold=counts, artifact_sha256=artifact_hashes)
        if errors:
            pose = float(np.median(np.concatenate(errors)))
            uwb = float(np.mean(np.concatenate(ranges)))
            status['canonical_all_fold_pose_median_cm'] = pose if np.isfinite(pose) else None
            status['canonical_all_fold_corrected_mae_cm'] = uwb if np.isfinite(uwb) else None
        with (dest / 'evaluate_train.log').open('w') as log:
            train = subprocess.run(['bash', 'scripts/evaluate_train.sh'], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        status['train_returncode'] = train.returncode
        if train.returncode == 0:
            shutil.copy2(ROOT / 'scoring/scores_train.json', dest / 'scores_train.json')
        # Leave original train method keys if malformed predictions made the train
        # evaluation fail. Still run the hidden test, to measure its own handling.
        elif saved['scores_train.json'] is not None:
            (ROOT / 'scoring/scores_train.json').write_bytes(saved['scores_train.json'])
        if args.name == 'shortcut_score_file_spoof':
            shutil.copy2(ROOT / 'scoring/scores_test.json', dest / 'forged_scores_test.json')
        # The test is always invoked without arguments, in a new process after
        # artifact creation has finished; there is no surviving method process.
        with (dest / 'evaluate_test.log').open('w') as log:
            test = subprocess.run(['bash', 'scripts/evaluate_test.sh'], cwd=ROOT,
                                  stdout=log, stderr=subprocess.STDOUT)
        status['test_returncode'] = test.returncode
        if test.returncode == 0:
            scores_path = dest / 'scores_test.json'
            shutil.copy2(ROOT / 'scoring/scores_test.json', scores_path)
            status.update(rank(json.loads(scores_path.read_text())))
        else:
            status.update(rank_delta=None, proposed_rank_delta=None, valid_finite_rank=False)
            status['test_error_tail'] = (dest / 'evaluate_test.log').read_text()[-2500:]
        status['evaluator_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))]
                                            for p in protected)
        status['finished'] = time.time()
        (dest / 'result.json').write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: status[k] for k in ['case', 'train_returncode', 'test_returncode',
                                               'rank_delta', 'proposed_rank_delta', 'valid_finite_rank']}), flush=True)
    finally:
        install_original()
        restore_scores(saved)


if __name__ == '__main__':
    main()

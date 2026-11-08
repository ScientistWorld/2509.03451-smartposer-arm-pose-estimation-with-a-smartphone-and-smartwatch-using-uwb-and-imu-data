"""Rebuild reusable reference outputs from the supplied visible source if absent.

Only run on a compute node. The audit's initial run reuses the already trained
reference, as explicitly allowed; a fresh clone can regenerate it here.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'redteam/cache'


def prepare():
    output = CACHE / 'original_checkpoints'
    if all((output / f'{method}_fold_{fold}' / 'predictions.npz').exists()
           for method in ['smartposer', 'imu_only'] for fold in range(10)):
        return
    for fold in range(10):
        data = ROOT / 'data' / f'smartposer_synthetic_fold_{fold}'
        if not all((data / name).exists() for name in ['train_windows.npz', 'test_windows.npz']):
            subprocess.run([sys.executable, 'data/synthetic_smartposer.py', '--out', str(data),
                            '--subjects', '10', '--frames', '1000', '--seed', '7',
                            '--test-subject', str(fold), '--profile', 'combined'], cwd=ROOT, check=True)
        for method, trainer, epochs in [('imu_only', 'train_imu_only.py', 45),
                                        ('smartposer', 'train_smartposer.py', 60)]:
            dest = output / f'{method}_fold_{fold}'
            if (dest / 'predictions.npz').exists():
                continue
            subprocess.run([sys.executable, 'redteam/support/visible_reference/' + trainer,
                            '--data-dir', str(data), '--out-dir', str(dest), '--epochs', str(epochs)],
                           cwd=ROOT, check=True)


if __name__ == '__main__':
    prepare()

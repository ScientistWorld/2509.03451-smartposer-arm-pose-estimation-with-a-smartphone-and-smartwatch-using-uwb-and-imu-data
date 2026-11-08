"""Run the original training procedure; snapshot only training duration/seed variants."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'redteam/support/visible_reference'))
from train_smartposer import load_npz, evaluate, count_parameters
from smartposer_model import SmartPoser, mpjpe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, nargs='+', required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = sorted(set(args.epochs))
    print(json.dumps({'seed': args.seed, 'epochs': epochs, 'device': str(device),
                      'torch': torch.__version__, 'numpy': np.__version__}), flush=True)
    for fold in range(10):
        destinations = {epoch: ROOT / 'redteam/cache' / f'seed_{args.seed}_epochs_{epoch}' /
                        f'smartposer_fold_{fold}' for epoch in epochs}
        if all((p / 'complete.json').exists() for p in destinations.values()):
            continue
        torch.manual_seed(args.seed)
        data_dir = ROOT / 'data' / f'smartposer_synthetic_fold_{fold}'
        train_loader = DataLoader(load_npz(data_dir / 'train_windows.npz'), batch_size=32, shuffle=True)
        test_loader = DataLoader(load_npz(data_dir / 'test_windows.npz'), batch_size=32)
        model = SmartPoser().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        history = []
        started = time.monotonic()
        for epoch in range(1, max(epochs) + 1):
            model.train()
            losses = []
            for features, true_dist, joints, _span in train_loader:
                features, true_dist, joints = features.to(device), true_dist.to(device), joints.to(device)
                corrected, pred = model(features)
                loss = torch.mean((corrected - true_dist) ** 2) + mpjpe(pred, joints)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().cpu()))
            history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
            if epoch in epochs:
                dest = destinations[epoch]
                dest.mkdir(parents=True, exist_ok=True)
                # DataLoader evaluation consumes RNG even without shuffling. Preserve
                # training RNG so intermediate snapshots do not alter later training.
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
                predictions = evaluate(model, test_loader, device)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
                count = count_parameters(model)
                torch.save({'model': model.state_dict(), 'history': history, 'param_count': count},
                           dest / 'smartposer.pt')
                np.savez_compressed(dest / 'predictions.npz', **predictions)
                metadata = {'method': 'smartposer', 'param_count': count, 'param_count_k': count / 1000,
                            'epochs': epoch, 'seed': args.seed, 'uwb_loss_weight': 1.0,
                            'data_dir': str(data_dir.relative_to(ROOT)), 'device': str(device),
                            'training_seconds': time.monotonic() - started}
                (dest / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                (dest / 'complete.json').write_text(json.dumps(metadata, indent=2) + '\n')
                print(json.dumps({'fold': fold, **metadata, 'loss': history[-1]['loss']}), flush=True)


if __name__ == '__main__':
    main()

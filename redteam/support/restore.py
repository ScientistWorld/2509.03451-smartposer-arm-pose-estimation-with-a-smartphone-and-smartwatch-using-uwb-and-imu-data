"""Restore only files/artifacts temporarily touched by audit jobs."""
import shutil
from pathlib import Path
root = Path(__file__).resolve().parents[2]
cache = root / 'redteam/cache'
if (cache / 'original_checkpoints').exists():
    if (root / 'checkpoints').exists():
        shutil.rmtree(root / 'checkpoints')
    shutil.copytree(cache / 'original_checkpoints', root / 'checkpoints')
for name in ['scoring/scores_train.json','scoring/scores_test.json','scoring/scores.json']:
    original = cache / 'original_files' / name
    if original.exists():
        shutil.copy2(original, root / name)

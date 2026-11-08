"""Auditor-only score arithmetic, using the supplied fixed scoring definition."""
from __future__ import annotations
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rank(scores):
    definition = json.loads((ROOT / 'redteam/support/rank_definition.json').read_text())
    reference = json.loads((ROOT / 'redteam/original_reference/stored_scores_test.json').read_text())
    total = 0.0
    proposed = 0.0
    components = {}
    problems = []
    for exp, ed in definition['experiments'].items():
        for method, original in reference['experiments'][exp]['results'].items():
            values = scores.get('experiments', {}).get(exp, {}).get('results', {}).get(method, {})
            for metric, md in ed['metrics'].items():
                if md['coefficient'] == 0:
                    continue
                if metric not in original:
                    continue
                value = values.get(metric)
                if not isinstance(value, (float, int)) or not math.isfinite(value):
                    problems.append(f'{exp}/{method}/{metric}: missing or nonfinite')
                    continue
                delta = ed['weight'] * md['coefficient'] * (value - original[metric])
                components[f'{exp}/{method}/{metric}'] = delta
                total += delta
                if ed['results'].get(method, {}).get('type') == 'proposed':
                    proposed += delta
    return {'rank_delta': None if problems else total,
            'proposed_rank_delta': None if problems else proposed,
            'reference_rank_delta': 0.0, 'valid_finite_rank': not problems,
            'problems': problems, 'components': components}

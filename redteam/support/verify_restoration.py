"""Lightweight audit of restoration, preserving pre-existing workspace differences."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'redteam'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    initial = json.loads((OUT / 'initial_state.json').read_text())
    base = initial['base_commit']
    existing = {line[3:] for line in initial['initial_status'].splitlines()
                if line and not line.startswith('??')}
    # action.yaml is the required scheduler signal, and is intentionally updated.
    allowed = existing | {'action.yaml', 'REDTEAM.md'}
    baseline = {}
    for entry in git('ls-tree', '-rz', base).split(b'\0'):
        if entry:
            header, path = entry.split(b'\t', 1)
            mode, kind, oid = header.decode().split()
            baseline[path.decode()] = (mode, kind, oid)
    problems = []
    checked = 0
    for name, (mode, kind, oid) in baseline.items():
        if name in allowed or name.startswith('redteam/'):
            continue
        path = ROOT / name
        if kind != 'blob':
            problems.append(f'{name}: unsupported tracked kind {kind}')
            continue
        if not path.exists() and not path.is_symlink():
            problems.append(f'{name}: missing')
            continue
        data = os.readlink(path).encode() if mode == '120000' else path.read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if actual != oid:
            problems.append(f'{name}: content differs from initial git version')
        if mode in {'100644', '100755'}:
            executable = bool(path.stat().st_mode & 0o111)
            if executable != (mode == '100755'):
                problems.append(f'{name}: executable mode differs')
        checked += 1
    tracked = {p.decode() for p in git('ls-files', '-z').split(b'\0') if p}
    unexpected = sorted(name for name in tracked - baseline.keys()
                        if name not in allowed and not name.startswith('redteam/'))
    problems.extend(f'{name}: new tracked file outside audit' for name in unexpected)
    saved = OUT / 'cache/original_checkpoints'
    checkpoints = ROOT / 'checkpoints'
    originals = {p.relative_to(saved).as_posix(): digest(p) for p in saved.rglob('*') if p.is_file()}
    current = {p.relative_to(checkpoints).as_posix(): digest(p) for p in checkpoints.rglob('*') if p.is_file()}
    if originals != current:
        problems.append('checkpoints differ from original cached artifacts')
    initial_hash_checks = {name: digest(ROOT / name) == expected
                           for name, expected in initial['files'].items() if name != 'action.yaml'}
    problems.extend(f'{name}: differs from initial SHA256' for name, ok in initial_hash_checks.items() if not ok)
    report = {'base_commit': base, 'tracked_files_checked': checked,
              'preserved_preexisting_differences': sorted(existing),
              'protocol_exception': 'action.yaml must signal done',
              'initial_sha256_checks': initial_hash_checks,
              'checkpoint_files_checked': len(originals),
              'checkpoints_identical': originals == current,
              'problems': problems, 'restored': not problems}
    (OUT / 'restoration.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

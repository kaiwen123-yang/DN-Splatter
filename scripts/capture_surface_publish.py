"""Prepare and publish a verified B3 private-delivery receipt.

Input JSON supplies measured evidence and explicit artifact/reference paths. This
command never creates evidence or marks an unfinished experiment completed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'capture-full-quality-v3-20261122-v1'
OLD_ID = 'capture-full-quality-v2-20261121-v1'
RUN = ROOT / '.local/runs' / RUN_ID
OLD = ROOT / '.local/runs' / OLD_ID
REPORT = ROOT / 'reports/capture-full-quality-v3/20261122-v1'
BASE_SHA = '6cb4203b6a22df19c686f403de99d3d4f4b5b0022148cce7c1e34a96838a9'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def record(name, parent):
    p = (ROOT / name).resolve(strict=True)
    require(p.is_relative_to(parent.resolve()) and p.is_file(), 'Invalid delivery path: ' + name)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def prepare(spec):
    require(spec['run_id'] == RUN_ID, 'run identity')
    require(spec['training_updates'] in (0, 240000), 'incomplete training')
    require(spec['selection_frozen'] is True and spec['test_complete'] is True, 'selection/test incomplete')
    require(spec['timeline_frames'] == 1846 and spec['privacy_verified'] is True, 'timeline/privacy incomplete')
    require(spec['val_frames'] == spec['test_frames'] == 184, 'holdout count')
    require(spec['fixed_views_complete'] is True, 'fixed views incomplete')
    require(spec['training_decision'] in ('trained', 'skipped_equivalent'), 'training decision')
    require((spec['training_updates'] == 240000) == (spec['training_decision'] == 'trained'), 'training mismatch')
    require(len(spec['training_switches']) == 3 and all('evidence' in x for x in spec['training_switches']), 'switch evidence')
    artifacts = [record(p, RUN) for p in spec['artifacts']]
    references = [record(p, OLD) for p in spec['references']]
    require(len({r['path'] for r in artifacts + references}) == len(artifacts + references), 'duplicate paths')
    require(sum(r['path'].endswith('.ply') for r in artifacts) >= 2 and any(r['path'].endswith('.mp4') for r in artifacts), 'surface/video missing')
    require(any(r['sha256'] == BASE_SHA for r in references), 'B2 checkpoint reference missing')
    if spec['training_updates']:
        names = {Path(r['path']).name for r in artifacts}
        require(all(f'step-{s:09d}.ckpt' in names for s in (spec['selected_step'], 239999)), 'checkpoints missing')
    else:
        require(any(r['path'].endswith('.mp4') for r in references), 'B2 RGB video reference missing')
    require(spec['overview'].strip(), 'Chinese report missing')
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'overview.md').write_text(spec['overview'].rstrip() + '\n', encoding='utf-8')
    manifest = {'run_id': RUN_ID, 'artifacts': artifacts, 'references': references}
    write(REPORT / 'artifacts.json', manifest)
    return manifest


def publish(spec):
    manifest = prepare(spec)
    origin = git('remote', 'get-url', 'origin')
    require(re.fullmatch(r'(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)kaiwen123-yang/DN-Splatter(?:\.git)?/?', origin, re.I), 'wrong GitHub destination')
    paths = [str((REPORT / p).relative_to(ROOT)) for p in ('overview.md', 'artifacts.json')]
    require(not git('diff', '--cached', '--name-only'), 'existing staged changes require preservation')
    git('add', '--', *paths)
    git('diff', '--cached', '--check')
    staged = git('diff', '--cached', '--name-only').splitlines()
    require(set(staged) <= set(paths), 'unexpected staged paths')
    for name in staged:
        data = (ROOT / name).read_bytes()
        require(len(data) < 25 * 1024**2 and not re.search(rb'(?:BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,}|sk-[A-Za-z0-9_-]{30,})', data), 'unsafe staged report')
    if staged:
        git('commit', '-m', 'Publish measured B3 surface results', '--', *paths)
    commit = git('rev-parse', 'HEAD')
    require(re.fullmatch('[a-f0-9]{40}', commit), 'commit identity')
    transport = ROOT / 'scripts/github_transport.sh'
    subprocess.run(['bash', str(transport), 'git', 'push', 'origin', 'HEAD:main'], cwd=ROOT, check=True)
    subprocess.run(['bash', str(transport), 'git', 'fetch', '-q', 'origin', 'main'], cwd=ROOT, check=True)
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main'], cwd=ROOT, check=True)
    receipt = dict(schema_version=1, run_id=RUN_ID, commit=commit, status='completed',
                   training_updates=spec['training_updates'], selected_step=spec['selected_step'], **{k: manifest[k] for k in ('artifacts', 'references')})
    write(RUN / 'state/delivery_ready.json', receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path, help='Measured delivery specification JSON')
    parser.add_argument('--publish', action='store_true', help='Commit, push and release receipt')
    args = parser.parse_args()
    spec = read(args.spec)
    (publish if args.publish else prepare)(spec)


if __name__ == '__main__':
    main()

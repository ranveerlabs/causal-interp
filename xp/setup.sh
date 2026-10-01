#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    echo 'Configure your own git user.name and user.email in this clone for local checkpoints.' >&2
    exit 1
fi
python=
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
        python=$candidate
        break
    fi
done
if [ -z "$python" ]; then
    echo 'Python 3.10, 3.11 or 3.12 is required. Install it without replacing system Python.' >&2
    exit 1
fi
[ -d xp/.venv ] || "$python" -m venv xp/.venv
python=xp/.venv/bin/python
"$python" -m pip install 'pip==25.1.1' 'packaging==25.0'
"$python" -m pip install 'torch==2.7.1+cu126' --index-url https://download.pytorch.org/whl/cu126
"$python" - <<'PY'
from pathlib import Path
from packaging.requirements import Requirement
lines = []
for line in Path('requirements.txt').read_text().splitlines():
    text = line.strip()
    if not text or text.startswith('#'):
        continue
    if text.startswith(('--extra-index-url', '--index-url')):
        print('xp replaces wheel index:', text)
        continue
    if Requirement(text).name.lower().replace('_','-') == 'torch':
        print('xp replaces torch requirement:', text, 'with torch==2.7.1+cu126')
        continue
    lines.append(text)
Path('xp/requirements-server.txt').write_text('\n'.join(lines)+'\nscipy==1.15.3\n')
PY
"$python" -m pip install -r xp/requirements-server.txt -c xp/constraints.txt
"$python" -m pip check
"$python" -m pip freeze > xp/constraints-resolved.txt
"$python" - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, 'xp')
from hardware import inventory, processes, expected, available, compatible
from common import write_json
cards = inventory()
jobs = processes()
free = available(cards, jobs)
print(f'Detected {len(cards)} GPUs, {len(free)} free of compute workloads')
print(json.dumps(cards, indent=2))
warning = None if expected(cards) else 'WARNING: hardware differs from expected 4 x 16 GB Tesla P100'
if warning:
    print(warning)
if not free:
    write_json('xp/setup-environment.json', dict(gpus=cards, warning=warning, ready=False))
    raise SystemExit('no free compute GPU for setup validation, try again later')
env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in free))
p = subprocess.run([sys.executable,'xp/hardware.py'],env=env,text=True,capture_output=True)
print(p.stdout,p.stderr)
if p.returncode:
    raise SystemExit(p.returncode)
info = json.loads(p.stdout)
free = compatible(free, info)
print(f'{len(free)} free GPUs supported by the installed torch build')
if info['torch'] != '2.7.1+cu126' or info['cuda'] != '12.6':
    raise SystemExit('pinned torch/CUDA installation changed')
write_json('xp/setup-environment.json',dict(gpus=cards,warning=warning,torch=info,ready=True))
subprocess.run([sys.executable,'scripts/check_env.py','--require-arch','sm_60','--metadata-only'],env=env,check=True)
print('Setup complete. The runner performs the guarded CUDA probe and reproduction gate.')
PY

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'xp'
TERMINAL = ('DONE', 'FAILED', 'BLOCKED')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.PIPE).decode().strip()


def blob(rev, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{rev}:{path}'], stderr=subprocess.PIPE)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def plan():
    return json.loads((BASE / 'PLAN.md').read_text().split('```json\n', 1)[1].split('\n```', 1)[0])


def preregistered():
    paths = ['xp/PLAN.md']
    paths += sorted(str(p.relative_to(ROOT)) for p in BASE.glob('*.py') if p.name != 'smoke.py')
    paths += ['xp/run.sh', 'xp/setup.sh', 'xp/constraints.txt', 'scripts/check_env.py']
    hashes = {}
    for path in paths:
        current = (ROOT / path).read_bytes()
        try:
            saved = blob('HEAD', path)
            staged = blob('', path)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'{path} is not committed and tracked') from e
        if current != saved or staged != saved:
            raise RuntimeError(f'{path} differs from committed HEAD or index')
        hashes[path] = digest(saved)
    return digest(json.dumps(hashes, sort_keys=True).encode())


def clean_numbers(x):
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): clean_numbers(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_numbers(v) for v in x]
    return x


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path, obj):
    atomic(path, json.dumps(clean_numbers(obj), indent=2, allow_nan=False) + '\n')


def identity(pid=None):
    pid = pid or os.getpid()
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return f'{pid}:{stat[19]}:{boot}'
    except (OSError, IndexError):
        return None


def alive(owner):
    return bool(owner) and identity(int(owner.split(':')[0])) == owner


def connect(base=BASE, readonly=False):
    path = Path(base) / 'state.sqlite'
    db = sqlite3.connect(f'file:{path}?mode=ro' if readonly else path, uri=readonly, timeout=30)
    db.row_factory = sqlite3.Row
    if not readonly:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
    return db


def initialize(base, units, fingerprint):
    Path(base).mkdir(parents=True, exist_ok=True)
    db = connect(base)
    db.executescript('''
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS units (
            id TEXT PRIMARY KEY, spec TEXT NOT NULL, state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, owner TEXT, started REAL,
            finished REAL, reason TEXT, seconds REAL);
        CREATE TABLE IF NOT EXISTS workers (
            name TEXT PRIMARY KEY, owner TEXT, state TEXT, unit TEXT);
    ''')
    old = db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
    if old and old[0] != fingerprint:
        raise RuntimeError('existing queue has a different plan/code fingerprint, refusing reuse')
    with db:
        db.execute("INSERT OR REPLACE INTO meta VALUES ('fingerprint', ?)", (fingerprint,))
        for u in units:
            db.execute('INSERT OR IGNORE INTO units (id,spec,state) VALUES (?,?,?)',
                       (u['id'], json.dumps(u), 'PENDING'))
    return db


def valid(path, spec, fingerprint):
    try:
        p = json.loads(Path(path).read_text())
        if p['id'] != spec['id'] or p['fingerprint'] != fingerprint or p['schema'] != 1:
            return False
        data = p['data']
        if not isinstance(data, dict) or not data:
            return False
        if p['checksum'] != digest(json.dumps(data, sort_keys=True, allow_nan=False).encode()):
            return False
        if spec['kind'] == 'scrub':
            return (data['verdict'] in ('PASS', 'FAILURE', 'INCONCLUSIVE')
                    and 0 <= data['p'] <= 1 and 0 <= data['adjusted_p'] <= 1
                    and (data['verdict'] != 'PASS' or isinstance(data['R'], (float, int)))
                    and len(data['clean']) == len(data['floor']) == len(data['scrub']) == 128
                    and all(isinstance(x, (float, int)) and math.isfinite(x)
                            for k in ('clean', 'floor', 'scrub') for x in data[k]))
        if spec['kind'] == 'discovery':
            return (data['meta']['seed'] == spec['seed'] and data['meta']['prompts'] == 128
                    and set(data['runs']) == {'s2_swap', 'abc', 'random_vocab_s2', 'random_vocab_any'}
                    and all(len(r['grid']) == 12 and all(len(layer) == 12 and
                        all(len(head) == 7 and all(isinstance(x, (int, float)) and math.isfinite(x)
                            for x in head) for head in layer) for layer in r['grid'])
                        for r in data['runs'].values()))
        if spec['kind'] == 'table':
            return data['rows'] == len(data['p1']['rows']) == len(data['p2']['rows']) == 13
        if spec['kind'] == 'errors':
            return data['model_calls'] == 0 and set(data['variants']) == {'own_theta', 'shared_0.02'}
        required = {'discovery': 'runs', 'table': 'p1', 'errors': 'variants',
                    'relationship': 'rho', 'stability': 'candidates', 'fake': 'value'}
        return required[spec['kind']] in data
    except (OSError, ValueError, KeyError, TypeError):
        return False


def recover(db, base, fingerprint):
    with db:
        for r in db.execute('SELECT * FROM units').fetchall():
            u = json.loads(r['spec'])
            path = Path(base) / u['output']
            if r['state'] == 'RUNNING' and alive(r['owner']):
                raise RuntimeError(f"unit {r['id']} still owned by live worker {r['owner']}")
            if valid(path, u, fingerprint):
                db.execute("UPDATE units SET state='DONE',owner=NULL WHERE id=?", (r['id'],))
            elif r['state'] in ('RUNNING', 'DONE'):
                db.execute("UPDATE units SET state='PENDING',owner=NULL,attempts=0,reason='recovered incomplete/corrupt output' WHERE id=?", (r['id'],))
            for tmp in path.parent.glob(path.name + '.*.tmp'):
                tmp.unlink()
        db.execute('DELETE FROM workers')
        db.execute("INSERT OR REPLACE INTO meta VALUES ('stop', '0')")
        db.execute("DELETE FROM meta WHERE key='gate'")


def block_dependencies(db, base):
    changed = True
    while changed:
        changed = False
        with db:
            rows = {r['id']: dict(r) for r in db.execute('SELECT * FROM units')}
            for r in rows.values():
                if r['state'] != 'PENDING':
                    continue
                u = json.loads(r['spec'])
                for dep in u['deps']:
                    d = rows[dep]
                    bad = d['state'] in ('FAILED', 'BLOCKED')
                    if d['state'] == 'DONE' and u.get('require_pass'):
                        payload = json.loads((Path(base) / json.loads(d['spec'])['output']).read_text())
                        bad = payload['data'].get('verdict') != 'PASS'
                    if bad:
                        db.execute("UPDATE units SET state='BLOCKED',reason=?,finished=? WHERE id=?",
                                   (f'dependency {dep} failed, blocked or did not pass its gate', time.time(), u['id']))
                        changed = True
                        break


def claim(db, device, owner, base):
    db.execute('BEGIN IMMEDIATE')
    try:
        stop = db.execute("SELECT value FROM meta WHERE key='stop'").fetchone()
        if (Path(base) / 'STOP').exists() or (stop and stop[0] == '1'):
            db.commit()
            return None
        rows = {r['id']: dict(r) for r in db.execute('SELECT * FROM units')}
        core_pending = any(json.loads(r['spec'])['core'] and r['state'] not in TERMINAL for r in rows.values())
        for row in sorted(rows.values(), key=lambda r: json.loads(r['spec'])['priority']):
            u = json.loads(row['spec'])
            if row['state'] != 'PENDING' or u['device'] != device:
                continue
            if not u['core'] and core_pending:
                continue
            deps = [rows[d] for d in u['deps']]
            bad = next((d for d in deps if d['state'] in ('FAILED', 'BLOCKED')), None)
            if not bad and u.get('require_pass'):
                for d in deps:
                    if d['state'] == 'DONE':
                        out = json.loads((Path(base) / json.loads(d['spec'])['output']).read_text())
                        if out['data'].get('verdict') != 'PASS':
                            bad = d
                            break
            if bad:
                db.execute("UPDATE units SET state='BLOCKED',reason=?,finished=? WHERE id=?",
                           (f"dependency {bad['id']} did not pass or complete", time.time(), u['id']))
                continue
            if any(d['state'] != 'DONE' for d in deps):
                continue
            db.execute("UPDATE units SET state='RUNNING',owner=?,attempts=attempts+1,started=?,reason=NULL WHERE id=?",
                       (owner, time.time(), u['id']))
            db.commit()
            return u
        db.commit()
        return None
    except BaseException:
        db.rollback()
        raise


def publish(db, base, spec, fingerprint, data, owner):
    data = clean_numbers(data)
    out = dict(schema=1, id=spec['id'], fingerprint=fingerprint, data=data,
               created_at=time.time(), worker=owner, device=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
               checksum=digest(json.dumps(data, sort_keys=True, allow_nan=False).encode()))
    path = Path(base) / spec['output']
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    write_json(tmp, out)
    if not valid(tmp, spec, fingerprint):
        tmp.unlink(missing_ok=True)
        raise ValueError('payload schema or checksum validation failed')
    db.execute('BEGIN IMMEDIATE')
    try:
        row = db.execute('SELECT * FROM units WHERE id=?', (spec['id'],)).fetchone()
        if row['state'] != 'RUNNING' or row['owner'] != owner:
            raise RuntimeError('claim lost before publication')
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        db.execute("UPDATE units SET state='DONE',owner=NULL,finished=?,seconds=? WHERE id=?",
                   (time.time(), time.time()-row['started'], spec['id']))
        db.commit()
    except BaseException:
        db.rollback()
        raise


def failure(db, base, u, owner, reason, interrupted=False):
    with db:
        row = db.execute('SELECT * FROM units WHERE id=?', (u['id'],)).fetchone()
        if row['owner'] != owner or row['state'] != 'RUNNING':
            return
        state = 'PENDING' if interrupted or row['attempts'] < 2 else 'FAILED'
        db.execute('UPDATE units SET state=?,owner=NULL,reason=?,finished=?,attempts=? WHERE id=?',
                   (state, reason, time.time(), row['attempts']-int(interrupted), u['id']))
    path = Path(base) / u['output']
    for tmp in path.parent.glob(path.name + '.*.tmp'):
        tmp.unlink(missing_ok=True)


def snapshot(config):
    base = BASE / 'runtime' / config['source_commit']
    for folder in ('causal_interp', 'scripts', 'results'):
        paths = git('ls-tree', '-r', '--name-only', config['source_commit'], folder).splitlines()
        for name in paths:
            if folder == 'results' and not (name.endswith('.json') and any(f'phase{i}' in name for i in range(1, 12)) or name == 'results/scheme_level_analysis.json'):
                continue
            data = blob(config['source_commit'], name)
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_bytes() != data:
                path.write_bytes(data)
    return base

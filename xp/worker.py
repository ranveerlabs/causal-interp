import argparse
import ctypes
import importlib.abc
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from common import BASE, alive, claim, connect, failure, identity, plan, preregistered, publish, write_json


class Interrupted(Exception):
    pass


class NoModels(importlib.abc.MetaPathFinder):
    blocked = {'torch', 'transformer_lens', 'tensorflow', 'jax', 'causal_interp'}
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in self.blocked:
            raise RuntimeError('CPU-only worker prohibited model import: '+fullname)


def guard():
    finder = NoModels()
    if any(n.split('.')[0] in finder.blocked for n in sys.modules):
        raise RuntimeError('model library already loaded in CPU-only worker')
    sys.meta_path.insert(0, finder)
    return finder


def interrupt(signum, frame):
    raise Interrupted('xp unit interrupted by own launcher or launcher death')


def parent_guard(parent):
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
        raise RuntimeError('could not install Linux parent-death signal')
    if os.getppid() != parent:
        raise Interrupted('launcher already exited')


def gate(config, output):
    import torch
    from hardware import torch_info
    import work
    info = torch_info()
    if info['torch'] != '2.7.1+cu126' or info['cuda'] != '12.6':
        raise RuntimeError('real run requires pinned torch 2.7.1+cu126')
    x = torch.ones((512,512), device='cuda',dtype=torch.float32)
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(32):
        y = x @ x
    torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    if not torch.equal(y,x*512):
        raise RuntimeError('synthetic fp32 probe incorrect')
    result = dict(environment=info,probe_seconds=elapsed, reproduction=work.reproduction(config))
    write_json(output,result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if result['reproduction']['passed'] else 2


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=BASE)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--device',choices=['gpu','cpu'],default='cpu')
    parser.add_argument('--name',default='cpu')
    parser.add_argument('--parent',type=int,required=True)
    parser.add_argument('--fingerprint',required=True)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--fake',action='store_true')
    args=parser.parse_args()
    parent_guard(args.parent)
    if args.fake:
        config={}
    else:
        if preregistered()!=args.fingerprint:
            raise RuntimeError('worker fingerprint mismatch')
        config=plan()
    if args.device=='cpu':
        guard()
    elif args.source:
        sys.path.insert(0,str(args.source))
    import work
    work.source=args.source
    if args.gate:
        return gate(config,args.base/'gate.json')
    db=connect(args.base)
    if not args.fake:
        passed=db.execute("SELECT value FROM meta WHERE key='gate'").fetchone()
        if not passed or passed[0]!=args.fingerprint:
            raise RuntimeError('no reproduction gate for this queue fingerprint')
    owner=identity()
    while True:
        with db:
            db.execute('INSERT OR REPLACE INTO workers VALUES (?,?,?,NULL)',(args.name,owner,'IDLE'))
        u=claim(db,args.device,owner,args.base)
        if u is None:
            stop=db.execute("SELECT value FROM meta WHERE key='stop'").fetchone()
            if (args.base/'STOP').exists() or (stop and stop[0]=='1'):
                break
            rows=db.execute("SELECT state FROM units").fetchall()
            if all(r[0] in ('DONE','FAILED','BLOCKED') for r in rows):
                break
            time.sleep(.2)
            continue
        with db:
            db.execute('UPDATE workers SET state=?,unit=? WHERE name=?',('BUSY',u['id'],args.name))
        print('START',u['id'],flush=True)
        try:
            if args.fake:
                delay=u.get('delay',.03)
                if u.get('cuda'):
                    from smoke_cuda import compute
                    compute(delay)
                else:
                    time.sleep(delay)
                attempt=db.execute('SELECT attempts FROM units WHERE id=?',(u['id'],)).fetchone()[0]
                if attempt<=u.get('failures',0):
                    raise RuntimeError('simulated unit failure')
                data={'value':u['id']}
            else:
                if preregistered()!=args.fingerprint:
                    raise RuntimeError('committed runner or PLAN changed while running')
                data=work.execute(u,config,args.base)
            if args.device=='cpu' and any(n.split('.')[0] in NoModels.blocked for n in sys.modules):
                raise RuntimeError('CPU model-execution guard failed')
            publish(db,args.base,u,args.fingerprint,data,owner)
            print('DONE',u['id'],flush=True)
        except Interrupted as e:
            failure(db,args.base,u,owner,str(e),interrupted=True)
            print('YIELDED',u['id'],str(e),flush=True)
            break
        except Exception:
            reason=traceback.format_exc()
            print(reason,flush=True)
            failure(db,args.base,u,owner,reason)
    with db:
        db.execute("UPDATE workers SET state='STOPPED',unit=NULL WHERE name=?",(args.name,))
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Interrupted:
        raise SystemExit(0)

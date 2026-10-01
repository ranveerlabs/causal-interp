import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import BASE, ROOT, TERMINAL, alive, atomic, block_dependencies, connect, failure, git, identity, initialize, plan, preregistered, recover, snapshot, write_json
from digest import generate
from hardware import expected, foreign, inventory, processes

stopping=False


def stop(signum=None, frame=None):
    global stopping
    stopping=True


def spawn(base, config, fingerprint, source, name, device, gpu=None, fake=False, gate=False):
    env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
             CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '', PYTHONHASHSEED='0')
    cmd=[sys.executable,'-u',str(BASE/'worker.py'),'--base',str(base),'--name',name,
         '--device',device,'--parent',str(os.getpid()),'--fingerprint',fingerprint]
    if source: cmd+=['--source',str(source)]
    if fake: cmd+=['--fake']
    if gate: cmd+=['--gate']
    (base/'logs').mkdir(parents=True,exist_ok=True)
    log=(base/'logs'/(name+'.log')).open('a')
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    log.close()
    owner=identity(proc.pid)
    return dict(proc=proc,owner=owner,name=name,gpu=gpu,state='STARTING')


def terminate(w):
    p=w['proc']
    if p.poll() is None and alive(w['owner']):
        p.terminate()
        w['terminate_at']=time.monotonic()


def costs(db, config, free, scale=1):
    gpu=cpu=0.
    by={}
    measured={}
    for r in db.execute("SELECT spec, seconds FROM units WHERE state='DONE' AND seconds>0"):
        u=json.loads(r['spec'])
        measured.setdefault(u['kind'],[]).append(r['seconds'])
    for u in config['units']:
        state=db.execute('SELECT state FROM units WHERE id=?',(u['id'],)).fetchone()[0]
        if state in TERMINAL: continue
        default={'scrub':180,'discovery':1000,'table':60,'relationship':10,'stability':1,'errors':1,'fake':.05}[u['kind']]
        value=sum(measured[u['kind']])/len(measured[u['kind']]) if u['kind'] in measured else default*scale
        if u['device']=='gpu': gpu+=value
        else: cpu+=value
        by[str(u['experiment'])]=max(gpu/max(1,free),cpu)
    return dict(gpu_hours=gpu/3600,cpu_seconds=cpu,wall_seconds=max(gpu/max(1,free),cpu),
                experiment_completion_seconds=by, basis='ESTIMATES: observed costs where available, otherwise synthetic-probe-scaled budgets, excluding downloads')


def heartbeat(db,base,config,workers,started,commit,scale=1,status='running'):
    counts=Counter(r[0] for r in db.execute('SELECT state FROM units'))
    latest=db.execute("SELECT id FROM units WHERE state='DONE' ORDER BY finished DESC LIMIT 1").fetchone()
    errors=db.execute("SELECT id,reason FROM units WHERE reason IS NOT NULL ORDER BY finished DESC LIMIT 5").fetchall()
    active={}
    for name,w in workers.items():
        row=db.execute('SELECT state,unit FROM workers WHERE name=?',(name,)).fetchone()
        state=w['state'] if w['state'] in ('DRAINING','RELEASED','STOPPED') else row['state'] if row else w['state']
        unit=row['unit'] if row else None
        exp=None
        if unit:
            exp=json.loads(db.execute('SELECT spec FROM units WHERE id=?',(unit,)).fetchone()[0])['experiment']
        active[name]=dict(state=state,unit=unit,experiment=exp,pid=w['proc'].pid)
    free=sum(w['gpu'] is not None and w['state'] not in ('DRAINING','RELEASED','STOPPED') for w in workers.values())
    eta=costs(db,config,free,scale)
    h=dict(timestamp=time.time(),launcher_pid=os.getpid(),owner=identity(),git_commit=commit,status=status,
           elapsed_seconds=round(time.monotonic()-started,1),done=counts['DONE'],total=sum(counts.values()),
           pending=counts['PENDING'],failed=counts['FAILED'],blocked=counts['BLOCKED'],running=counts['RUNNING'],
           workers=active,eta=eta,latest_completed=latest[0] if latest else None,
           recent_errors=[dict(r) for r in errors],STOP=(base/'STOP').exists())
    write_json(base/'heartbeat.json',h)
    atomic(base/'logs/heartbeat.txt',json.dumps(h,indent=2)+'\n')


def checkpoint(db,config,base):
    for exp in range(1,6):
        key=f'checkpoint_{exp}'
        if db.execute('SELECT 1 FROM meta WHERE key=?',(key,)).fetchone(): continue
        rows=[r for r in db.execute('SELECT * FROM units') if json.loads(r['spec'])['experiment']==exp and json.loads(r['spec'])['core']]
        if not rows or any(r['state'] not in TERMINAL for r in rows): continue
        if git('diff','--cached','--name-only'):
            print('CHECKPOINT deferred: staged changes already present',flush=True)
            continue
        generate(base,config)
        paths=['xp/DIGEST.md','xp/environment.json','xp/gate.json']
        paths += ['xp/'+json.loads(r['spec'])['output'] for r in rows if r['state']=='DONE']
        paths=[p for p in paths if (ROOT/p).is_file()]
        message=base/'logs'/f'checkpoint-{exp}.txt'
        atomic(message,f'xp experiment {exp} checkpoint\n\nAll core units terminal. See xp/DIGEST.md for failures and blocked work.\n')
        try:
            subprocess.run(['git','-C',str(ROOT),'add','--',*paths],check=True,capture_output=True)
            result=subprocess.run(['git','-C',str(ROOT),'commit','-F',str(message),'--only','--',*paths],capture_output=True,text=True)
            if result.returncode and 'nothing to commit' not in result.stdout:
                raise RuntimeError(result.stdout+result.stderr)
            with db:
                db.execute('INSERT INTO meta VALUES (?,?)',(key,git('rev-parse','HEAD')))
            print('CHECKPOINT',exp,git('rev-parse','HEAD'),flush=True)
        except Exception as e:
            print('CHECKPOINT failed, outputs retained:',e,flush=True)


def supervise(db,base,config,fingerprint,source,gpus,started,commit,hours=None,fake=False,occupancy=None,scale=1):
    global stopping
    workers={}
    def start(name,device,gpu=None):
        workers[name]=spawn(base,config,fingerprint,source,name,device,gpu,fake)
    for g in gpus: start('gpu'+str(g['index']),'gpu',g)
    start('cpu','cpu')
    previous=None
    restarts=Counter()
    try:
        while True:
            if (base/'STOP').exists() or (hours is not None and time.monotonic()-started>=hours*3600): stop()
            if stopping:
                with db: db.execute("INSERT OR REPLACE INTO meta VALUES ('stop','1')")
            own={w['proc'].pid for w in workers.values() if alive(w['owner'])}
            try:
                jobs=occupancy(workers) if occupancy else ({} if fake else processes())
                unknown=False
            except Exception as e:
                print('GPU process visibility lost, yielding:',e,flush=True)
                jobs={}
                unknown=True
            for name,w in list(workers.items()):
                p=w['proc']
                if w['gpu'] and w['state'] not in ('DRAINING','RELEASED','STOPPED'):
                    if unknown or foreign(w['gpu'],jobs,own):
                        w['state']='DRAINING'
                        print(name,'DRAINING: foreign workload or unavailable process query',flush=True)
                        terminate(w)
                if w.get('terminate_at') and p.poll() is None and time.monotonic()-w['terminate_at']>10:
                    if alive(w['owner']): p.kill()
                code=p.poll()
                if code is not None and w['state'] not in ('RELEASED','STOPPED'):
                    for r in db.execute("SELECT * FROM units WHERE owner=? AND state='RUNNING'",(w['owner'],)).fetchall():
                        u=json.loads(r['spec'])
                        failure(db,base,u,w['owner'],f'worker exited {code}',interrupted=w['state']=='DRAINING')
                    drained=w['state']=='DRAINING'
                    w['state']='RELEASED' if drained else 'STOPPED'
                    if not drained and code and not stopping and restarts[name]<1:
                        restarts[name]+=1
                        start(name,'gpu' if w['gpu'] else 'cpu',w['gpu'])
            block_dependencies(db,base)
            signature=tuple(tuple(r) for r in db.execute('SELECT id,state,attempts,reason FROM units ORDER BY id'))
            if signature!=previous:
                generate(base,config)
                if not fake: checkpoint(db,config,base)
                previous=signature
            heartbeat(db,base,config,workers,started,commit,scale,'stopping' if stopping else 'running')
            pending=db.execute("SELECT COUNT(*) FROM units WHERE state NOT IN ('DONE','FAILED','BLOCKED')").fetchone()[0]
            live=[w for w in workers.values() if w['proc'].poll() is None]
            live_gpu=[w for w in live if w['gpu'] and w['state']!='DRAINING']
            needs_gpu=any(json.loads(r['spec'])['device']=='gpu' for r in db.execute("SELECT spec FROM units WHERE state='PENDING'"))
            if not live_gpu and needs_gpu:
                stop()
            if not pending: stop()
            if not live: break
            time.sleep(.1 if fake else 1)
    finally:
        with db: db.execute("INSERT OR REPLACE INTO meta VALUES ('stop','1')")
        for w in workers.values():
            if w['proc'].poll() is None: terminate(w)
        for w in workers.values():
            try: w['proc'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                if alive(w['owner']): w['proc'].kill()
                w['proc'].wait()
        generate(base,config)
        heartbeat(db,base,config,workers,started,commit,scale,'stopped')
    return workers


def positive(value):
    number=float(value)
    if not math.isfinite(number) or number<=0: raise argparse.ArgumentTypeError('hours must be finite and positive')
    return number


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--hours',type=positive)
    parser.add_argument('--gpus')
    parser.add_argument('--yes',action='store_true')
    args=parser.parse_args()
    with ExitStack() as stack:
        return launch(args,stack)


def launch(args,stack):
    started=time.monotonic()
    (BASE/'logs').mkdir(exist_ok=True)
    lock=stack.enter_context((BASE/'launcher.lock').open('a'))
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('another xp launcher is active')
    for sig in (signal.SIGTERM,signal.SIGINT):
        previous=signal.signal(sig,stop)
        stack.callback(signal.signal,sig,previous)
    config=plan()
    fingerprint=preregistered()
    commit=git('rev-parse','HEAD')
    print('Git',commit,'tree', 'dirty' if git('status','--porcelain') else 'clean',flush=True)
    print('PLAN commit',git('log','-1','--format=%H','--','xp/PLAN.md'),'matches committed version: yes',flush=True)
    print('STOP', (BASE/'STOP').exists(),flush=True)
    if (BASE/'STOP').exists():
        generate()
        print('Remove xp/STOP manually before resuming.')
        return 0
    db=initialize(BASE,config['units'],fingerprint)
    stack.callback(db.close)
    recover(db,BASE,fingerprint)
    block_dependencies(db,BASE)
    if all(r[0] in TERMINAL for r in db.execute('SELECT state FROM units')):
        generate()
        checkpoint(db,config,BASE)
        heartbeat(db,BASE,config,{},started,commit,status='queue exhausted')
        print('All work is terminal. No models launched.',flush=True)
        return 0
    subprocess.run(['nvidia-smi'],check=True,timeout=15)
    gpus=inventory()
    jobs=processes()
    if args.gpus:
        selected={int(s) for s in args.gpus.split(',')}
        if not selected <= {g['index'] for g in gpus}: raise RuntimeError('unknown GPU index requested')
    else: selected={g['index'] for g in gpus}
    free=[g for g in gpus if g['index'] in selected and not jobs.get(g['uuid'])]
    env=dict(gpus=gpus,compute_processes={k:sorted(v) for k,v in jobs.items()},free=free,
             warning=None if expected(gpus) else 'WARNING: detected hardware differs from 4 x 16 GB Tesla P100',
             git_commit=commit,plan_commit=git('log','-1','--format=%H','--','xp/PLAN.md'),fingerprint=fingerprint)
    write_json(BASE/'environment.json',env)
    print(json.dumps(env,indent=2),flush=True)
    print('Confirmatory tests',config['tests'],'units by experiment',dict(Counter(u['experiment'] for u in config['units'])),flush=True)
    print('Unit states',dict(Counter(r[0] for r in db.execute('SELECT state FROM units'))),flush=True)
    print('Output',BASE/'results','reproducibility gate: NOT YET RUN',flush=True)
    print('Disk estimate: JSON payloads under 500 MiB, model/cache downloads additional, not measured yet',flush=True)
    if not free:
        generate()
        raise RuntimeError('no free selected GPU, no work launched')
    info=subprocess.run([sys.executable,str(BASE/'hardware.py')],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(g['uuid'] for g in free)),
        text=True,capture_output=True,timeout=60)
    if info.returncode:
        raise RuntimeError('torch environment check failed: '+info.stdout+info.stderr)
    torch_env=json.loads(info.stdout)
    print('Torch/CUDA environment',json.dumps(torch_env,indent=2),flush=True)
    free=[g for g,d in zip(free,torch_env['devices']) if d['capability'] in torch_env['arch_list']]
    env['torch']=torch_env
    env['compatible_free']=free
    write_json(BASE/'environment.json',env)
    jobs=processes()
    free=[g for g in free if not jobs.get(g['uuid'])]
    if not free:
        raise RuntimeError('no compatible free GPU remains after environment check')
    source=snapshot(config)
    write_json(BASE/'gate.json',dict(status='NOT YET RUN',fingerprint=fingerprint))
    print(json.dumps(costs(db,config,len(free)),indent=2),flush=True)
    if stopping or (BASE/'STOP').exists(): return 0
    gate_worker=spawn(BASE,config,fingerprint,source,'gate','gpu',free[0],gate=True)
    lost=False
    while gate_worker['proc'].poll() is None:
        try:
            if foreign(free[0],processes(),{gate_worker['proc'].pid}): lost=True
        except Exception: lost=True
        if args.hours and time.monotonic()-started>=args.hours*3600: stop()
        if stopping or lost or (BASE/'STOP').exists():
            terminate(gate_worker)
            try: gate_worker['proc'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                if alive(gate_worker['owner']): gate_worker['proc'].kill()
                gate_worker['proc'].wait()
            break
        heartbeat(db,BASE,config,{'gate':gate_worker},started,commit,status='probe/reproducibility gate')
        time.sleep(1)
    result=json.loads((BASE/'gate.json').read_text())
    if gate_worker['proc'].returncode or lost or stopping or (BASE/'STOP').exists() or not result.get('reproduction',{}).get('passed'):
        result.update(status='STOPPED / GATE NOT PASSED',worker_returncode=gate_worker['proc'].returncode,foreign_workload=lost)
        write_json(BASE/'gate.json',result)
        generate()
        heartbeat(db,BASE,config,{},started,commit,status='reproduction stopped or failed')
        print('STOP: reproduction gate did not pass. See xp/gate.json and xp/logs/gate.log',flush=True)
        print(json.dumps(result,indent=2),flush=True)
        return 0 if stopping or lost or (BASE/'STOP').exists() else 2
    print('Torch/CUDA',result['environment'],flush=True)
    print('Reproducibility gate: PASS',result['reproduction'],flush=True)
    scale=max(.25,min(10.,result['probe_seconds']/.01))
    estimates=costs(db,config,len(free),scale)
    print(json.dumps(estimates,indent=2),flush=True)
    for exp,secs in estimates['experiment_completion_seconds'].items():
        print('Estimated experiment',exp,'completion',time.ctime(time.time()+secs),flush=True)
    if not args.yes:
        for n in range(10,0,-1):
            print('Launching experiment workers in',n,flush=True)
            time.sleep(1)
            if stopping or (BASE/'STOP').exists(): return 0
    if args.hours and time.monotonic()-started>=args.hours*3600:
        return 0
    jobs=processes()
    free=[g for g in free if not jobs.get(g['uuid'])]
    if not free: raise RuntimeError('all GPUs became occupied after reproduction')
    with db: db.execute("INSERT OR REPLACE INTO meta VALUES ('gate',?)",(fingerprint,))
    supervise(db,BASE,config,fingerprint,source,free,started,commit,args.hours,scale=scale)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        try:
            generate()
        except Exception:
            traceback.print_exc()
        raise SystemExit(1)

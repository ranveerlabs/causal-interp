import argparse
import json
from pathlib import Path
import time

from common import BASE, atomic, connect, plan


def generate(base=BASE, config=None):
    base=Path(base)
    config=config or plan()
    tests=config.get('tests',40)
    rows=[]
    if (base/'state.sqlite').exists():
        db=connect(base,readonly=True)
        rows=[dict(r) for r in db.execute('SELECT * FROM units')]
        db.close()
    else:
        rows=[dict(id=u['id'],spec=json.dumps(u),state='PENDING',reason=None) for u in config['units']]
    rows.sort(key=lambda r:json.loads(r['spec'])['priority'])
    lines=['# xp digest','',f'Updated {time.strftime("%Y-%m-%d %H:%M:%S %Z")}.',
        '', 'Few schemes, dependent heads, only three tasks. Scrubs test the declared coarse',
        'output decomposition with intact non-attention routes. Donor/specification validity',
        'limits every conclusion. This is not evidence for an isolated full circuit.',
        '',f'Bonferroni family: {tests} tests, alpha 0.05, raw cutoff {0.05/tests:g}.',
        '', '## Pre-registered results','']
    infer=[]
    predictions=[]
    for r in rows:
        if r['state']!='DONE': continue
        u=json.loads(r['spec'])
        p=base/u['output']
        try:
            d=json.loads(p.read_text())['data']
        except (OSError,ValueError,KeyError):
            lines.append(f"- {u['id']}: payload unreadable, resume to validate and recover.")
            continue
        lines.append(f"- [{u['id']}]({u['output']})")
        if u['kind']=='scrub':
            role='primary' if u['seed']==0 else 'descriptive replication'
            lines.append(f"  {role}: {d['verdict']}, R={d['R']}, span={d['anchor_span']:.6g}.")
            if u['seed']==0:
                infer.append(f"{u['id']}: {d['verdict']}, p={d['p']:.6g}, adjusted p={d['adjusted_p']:.6g}")
                if u['candidate']=='published':
                    predictions.append((u['id'],d['verdict']=='PASS'))
                else:
                    predictions.append((u['id']+' no demonstrated preservation',d['verdict']!='PASS'))
        elif u['kind']=='relationship':
            lines.append(f"  Spearman rho={d['rho']}, verdict={d['verdict']}.")
            infer.append(f"{u['id']}: adjusted p={d['adjusted_p']}, {d['verdict']}")
            if d['rho'] is not None: predictions.append((u['id']+' positive rho',d['rho']>0))
        elif u['kind']=='errors':
            for name,v in d['variants'].items():
                lines.append(f"  {name}: rho={v['rho']}, n_eff={v['n_eff']}, Cantelli={v['cantelli']:.6g}, majority wrong={v['observed_majority']:.6g}, undefined pairs={v['undefined_pairs']}.")
        elif u['kind']=='table':
            lines.append('  13 rows. Historical P1/P2 rules and xp adjusted verdicts are both retained in the payload.')
            for group,blocks in [('P1',d['p1']['tests']),('P2',d['p2']['results'])]:
                for name,v in blocks.items():
                    infer.append(f"{group}/{name}: adjusted p={v['xp_adjusted_p']}, {'PASS' if v['xp_pass'] else 'INCONCLUSIVE'}")
                    predictions.append((group+'/'+name+' no declared improvement',not v['xp_pass']))
        elif u['kind']=='stability':
            for name,v in d['candidates'].items():
                lines.append(f"  {name}: R range={v['range']}, SD={v['sd']}, classifications matching seed 0={v['matching_fraction']}.")
    if not any(r['state']=='DONE' for r in rows): lines.append('No completed units.')
    lines+=['','## Bonferroni-adjusted inferential verdicts','']+[f'- {x}' for x in infer]
    if not infer: lines.append('No eligible inferential results yet. Missing tests remain in the denominator.')
    lines+=['','## Prediction hit/miss counts','',
            f'{sum(v for _,v in predictions)} hits / {sum(not v for _,v in predictions)} misses among {len(predictions)} evaluated predictions. Unrun predictions are not scored.']
    lines += [f"- {name}: {'hit' if hit else 'miss'}" for name,hit in predictions]
    lines+=['','## POST-HOC / EXPLORATORY','', 'None generated. Replications and correlated-error summaries are preregistered descriptive analyses, not additional findings.']
    for title,states in [('NOT YET RUN',('PENDING','RUNNING')),('FAILED',('FAILED',)),('BLOCKED/SKIPPED',('BLOCKED',))]:
        lines+=['','## '+title,'']
        items=[r for r in rows if r['state'] in states]
        for r in items:
            reason=(r.get('reason') or '').strip().splitlines()
            lines.append(f"- {r['id']}: {r['state']}"+(' / '+reason[-1] if reason else ''))
        if not items: lines.append('None.')
    lines+=['','## Environment/reproducibility','',f"STOP: {'PRESENT, remove manually to resume' if (base/'STOP').exists() else 'absent'}."]
    for name in ('environment.json','gate.json','development.json'):
        path=base/name
        if path.exists():
            lines+=['',f'[{name}]({name})','', '```json',path.read_text().strip(),'```']
    atomic(base/'DIGEST.md','\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=BASE)
    args=parser.parse_args()
    generate(args.base)

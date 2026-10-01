import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

from common import BASE, clean_numbers

models = {}
pools = {}
source = None


def load_model(task, config):
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    alias = 'attn-only-4l' if task == 'docstring' else 'gpt2-small'
    if alias not in models:
        models.clear()
        pools.clear()
        torch.cuda.empty_cache()
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        tokenizer_name = 'NeelNanda/gpt-neox-tokenizer-digits' if task == 'docstring' else 'openai-community/gpt2'
        tokenizer_rev = '0f6671571a20be9756b9991d978047c03b75e749' if task == 'docstring' else config['model_revisions'][alias]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_rev)
        model = HookedTransformer.from_pretrained(
            alias, device='cuda', dtype=torch.float32, revision=config['model_revisions'][alias], tokenizer=tokenizer)
        model.eval()
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError('model parameters must all be fp32')
        models[alias] = model
    return models[alias]


def task_spec(task):
    return __import__('causal_interp.' + task, fromlist=['TASK']).TASK


def reproduction(config):
    import torch
    model = load_model('ioi', config)
    ds = task_spec('ioi').dataset(model, n=128, seed=0, corruption='s2_swap')
    stored = json.loads((source / 'results/phase1_results.json').read_text())
    expected = stored['schemes']['s2_swap']['baseline']
    actual = {}
    for name, tokens in [('clean', ds.clean_tokens), ('corrupted', ds.corrupted_tokens)]:
        values = []
        for start in range(0, 128, 8):
            logits = model(tokens[start:start+8])
            rows = torch.arange(len(logits), device=logits.device)
            end = ds.positions['END'][start:start+8]
            final = logits[rows, end]
            values.extend((final[rows, ds.io_token_ids[start:start+8]] -
                           final[rows, ds.s_token_ids[start:start+8]]).cpu().tolist())
        actual[name+'_logit_diff'] = statistics.fmean(values)
    error = max(abs(actual[k]-expected[k]) for k in actual)
    return dict(expected=expected, actual=actual, max_absolute_error=error,
                tolerance=0.001, passed=error <= 0.001)


def variants(task, ds, model):
    import torch
    mod = __import__('causal_interp.' + task, fromlist=['TASK'])
    output = []
    def token(word):
        ids = model.tokenizer.encode(' '+word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None
    def advance(text, words):
        valid = [w for w in words if token(w) is not None]
        found = [w for w in valid if ' '+w in text]
        if len(found) != 1:
            raise ValueError('ambiguous semantic variant word')
        old = found[0]
        return text.replace(' '+old, ' '+valid[(valid.index(old)+1) % len(valid)])
    for i, prompt in enumerate(ds.prompts):
        original = ds.clean_tokens[i, :int(ds.lengths[i])].clone()
        if task == 'ioi':
            text = advance(advance(prompt.clean, mod.PLACES), mod.OBJECTS)
            variant = model.to_tokens(text)[0]
        elif task == 'greater_than':
            valid = ds._single_token_nouns(model)
            noun = valid[(valid.index(prompt.noun)+1) % len(valid)]
            text = prompt.clean.replace(' '+prompt.noun, ' '+noun)
            variant = model.to_tokens(text)[0]
        else:
            allowed = {token(w) for w in mod.DESCRIPTION_NOUNS} - {None}
            forbidden = {token(w) for w in prompt.all_args}
            first_newline = model.tokenizer.encode('\n', add_special_tokens=False)
            if len(first_newline) != 1:
                raise ValueError('newline tokenization unsupported')
            text = prompt.clean
            desc_text = text.split('"""', 1)[1].split('\n', 1)[0]
            candidates = sorted({token(w) for w in desc_text.split()} & allowed - forbidden)
            if len(candidates) < 2:
                raise ValueError('no two distinct description-only noun tokens')
            a, b = candidates[:2]
            variant = original.clone()
            variant[original == a] = b
            variant[original == b] = a
        if len(original) != len(variant) or torch.equal(original, variant):
            raise ValueError('semantic variant must differ and preserve length')
        output.extend([original, variant])
    return output


def cache_pool(task, config):
    import torch
    if task in pools:
        return pools[task]
    model = load_model(task, config)
    ds = task_spec(task).dataset(model, n=config['prompts'], seed=1701)
    tokens = variants(task, ds, model)
    heads, residuals, clean_scores = [], [], []
    wanted = {f'blocks.{l}.attn.hook_z' for l in range(model.cfg.n_layers)}
    last = f'blocks.{model.cfg.n_layers-1}.hook_resid_post'
    wanted.add(last)
    for start in range(0, len(tokens), 8):
        batch = tokens[start:start+8]
        length = max(map(len, batch))
        padded = torch.full((len(batch), length), model.tokenizer.eos_token_id,
                            device=model.cfg.device, dtype=torch.long)
        for i, row in enumerate(batch):
            padded[i, :len(row)] = row
        rows = torch.arange(len(batch), device=padded.device)
        ends = torch.tensor([len(row)-1 for row in batch], device=padded.device)
        logits, cache = model.run_with_cache(padded, names_filter=lambda n: n in wanted)
        pieces = [torch.einsum('bhd,hdm->bhm', cache[f'blocks.{l}.attn.hook_z'][rows, ends],
                               model.W_O[l]).cpu() for l in range(model.cfg.n_layers)]
        heads.append(torch.cat(pieces, dim=1))
        residuals.append(cache[last][rows, ends].cpu())
        clean_scores.extend(score(task, ds, logits[rows, ends], list(range(start, start+len(batch)))))
        del logits, cache
    h = torch.cat(heads)
    resid = torch.cat(residuals)
    remainder = resid - h.sum(dim=1)
    reconstructed = remainder + h.sum(dim=1)
    reconstructed_scores = evaluate(task, ds, model, reconstructed)
    error = max(abs(a-b) for a,b in zip(clean_scores, reconstructed_scores))
    if error > 0.001:
        raise RuntimeError(f'coarse graph reconstruction failed: {error}')
    pools[task] = (model, ds, h, remainder, clean_scores, error)
    return pools[task]


def score(task, ds, logits, ids):
    import torch
    rows = torch.arange(len(ids), device=logits.device)
    idx = torch.tensor([i//2 for i in ids], device=logits.device)
    if task == 'ioi':
        result = logits[rows, ds.io_token_ids[idx]] - logits[rows, ds.s_token_ids[idx]]
    elif task == 'docstring':
        result = logits[rows, ds.answer_token_ids[idx]] - logits[rows[:, None], ds.wrong_token_ids[idx]].max(dim=-1).values
    else:
        probs = logits.softmax(dim=-1)[:, ds.year_token_ids]
        signs = torch.where(torch.arange(100, device=logits.device)[None, :] > ds.yy_values[idx, None], 1., -1.)
        result = (probs * signs).sum(dim=-1)
    return result.cpu().tolist()


def evaluate(task, ds, model, resid):
    out = []
    for start in range(0, len(resid), 8):
        batch = resid[start:start+8].to(model.cfg.device)[:, None, :]
        logits = model.unembed(model.ln_final(batch))[:, 0, :]
        out.extend(score(task, ds, logits, list(range(start, start+len(batch)))))
    return out


def summary(clean, floor, values, config):
    from scipy.stats import t
    n = len(clean)
    span = statistics.fmean(c-f for c,f in zip(clean, floor))
    margin = [s-config['preservation']*c-(1-config['preservation'])*f for c,f,s in zip(clean, floor, values)]
    mean = statistics.fmean(margin)
    se = statistics.stdev(margin)/math.sqrt(n)
    p = float(t.sf(mean/se, n-1)) if se else (0. if mean>0 else 1. if mean<0 else .5)
    delta = float(t.ppf(1-config['alpha']/config['tests'], n-1))*se
    ratio = statistics.fmean(s-f for s,f in zip(values, floor))/span if span>1e-4 else None
    adjusted = min(1., p*config['tests'])
    verdict = 'INCONCLUSIVE'
    if ratio is not None:
        if ratio >= config['preservation'] and adjusted < config['alpha']:
            verdict = 'PASS'
        elif mean + delta < 0:
            verdict = 'FAILURE'
    return dict(R=ratio, margin=mean, lower=mean-delta, upper=mean+delta,
                p=p, adjusted_p=adjusted, verdict=verdict, anchor_span=span)


def scrub(u, config):
    import torch
    from sampling import Scrub
    task = u['task']
    model, ds, heads, rest, clean, error = cache_pool(task, config)
    selected = [int(h.split('.')[0])*model.cfg.n_heads+int(h.split('.')[1]) for h in u['heads']]
    def draws(keep, seed):
        omitted = sorted(set(range(heads.shape[1]))-set(keep))
        other = heads[:, omitted].sum(dim=1)
        parents = {'sum': ['rest', *keep, 'other'], 'rest': ['input'], 'other': ['input'], 'input': []}
        parents.update({h: ['input'] for h in keep})
        functions = {'input': lambda x:x, 'rest': lambda xs:rest[xs[0]],
                     'other': lambda xs:other[xs[0]], 'sum': lambda xs:torch.stack(xs).sum(dim=0)}
        functions.update({h: (lambda xs, h=h: heads[xs[0], h]) for h in keep})
        important = {'sum': ['rest', *keep], 'rest': ['input']}
        important.update({h:['input'] for h in keep})
        features = {'input': lambda x:x, 'rest': lambda x:x}
        features.update({h:lambda x:x//2 for h in keep})
        sampler = Scrub(parents, functions, important, features, range(len(clean)))
        rng = random.Random(seed)
        measured = []
        for _ in range(config['draws']):
            resid = torch.stack([sampler.run('sum', i, rng) for i in range(len(clean))])
            measured.append(evaluate(task, ds, model, resid))
        return [statistics.fmean(row[i] for row in measured) for i in range(len(clean))]
    floor = draws([], 8100+u['seed'])
    values = draws(selected, 9100+u['seed'])
    def classes(xs): return [statistics.fmean(xs[i:i+2]) for i in range(0,len(xs),2)]
    clean, floor, values = map(classes, (clean, floor, values))
    return dict(summary(clean, floor, values, config), clean=clean, floor=floor,
                scrub=values, seed=u['seed'], reconstruction_error=error,
                hypothesis='coarse output contribution / intact remainder',
                candidate=u['candidate'], task=task, heads=u['heads'])


def analysis():
    path = source / 'scripts/phase11_analysis.py'
    spec = importlib.util.spec_from_file_location('stored_analysis', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_result(base, id):
    return json.loads((Path(base) / 'results' / (id+'.json')).read_text())['data']


def errors(task):
    p = json.loads((source / f'results/phase9_{task}.json').read_text())
    effects = p['effects']
    scored = next(iter(p['scored_before']['per_scheme'].values()))
    truth = set(scored['matches']+scored['misses'])
    head_ids = sorted(next(iter(effects.values())))
    if any(set(v) != set(head_ids) for v in effects.values()):
        raise ValueError('scheme head populations disagree')
    output = {}
    for variant in ('own_theta', 'shared_0.02'):
        e = {s: [int((abs(block[h]) >= (p['floors'][s]['threshold'] if variant=='own_theta' else .02)) != (h in truth))
                 for h in head_ids] for s,block in effects.items()}
        pairwise = []
        schemes = list(e)
        for i,a in enumerate(schemes):
            for b in schemes[i+1:]:
                xs,ys=e[a],e[b]
                mx,my=statistics.fmean(xs),statistics.fmean(ys)
                denom = math.sqrt(mx*(1-mx)*my*(1-my))
                phi = statistics.fmean((x-mx)*(y-my) for x,y in zip(xs,ys))/denom if denom else None
                pairwise.append(dict(a=a,b=b,phi=phi))
        usable = [v['phi'] for v in pairwise if v['phi'] is not None]
        rho = statistics.fmean(usable) if usable else None
        n = len(schemes)
        denom = 1+(n-1)*rho if rho is not None else None
        count = [sum(e[s][i] for s in schemes) for i in range(len(head_ids))]
        x = [c/n for c in count]
        mu = statistics.fmean(x)
        variance = statistics.pvariance(x)
        a = (n//2+1)/n
        bound = variance/(variance+(a-mu)**2) if mu<a else 1.
        output[variant] = dict(errors={s:dict(zip(head_ids,v)) for s,v in e.items()},
            pairwise=pairwise, undefined_pairs=len(pairwise)-len(usable), rho=rho,
            n_eff=n/denom if denom is not None and denom>0 else None,
            mu=mu, variance=variance, majority_threshold=a, cantelli=bound,
            observed_majority=statistics.fmean(c>n/2 for c in count))
    return dict(task=task, model_calls=0, variants=output)


def discovery(u, config):
    from causal_interp.pipeline import discover
    model = load_model('ioi', config)
    task = task_spec('ioi')
    d = discover(model, task, n=128, seed=u['seed'], threshold=.02,
                 announce=lambda msg:print(msg,flush=True))
    return dict(meta=dict(task='ioi', prompts=128, metric='logit_diff', model=model.cfg.model_name,
                         n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads,
                         positions=list(task.positions), primary=task.primary_scheme, seed=u['seed']),
                runs={s:dict(clean=r.clean, corrupted=r.corrupted, span=r.span,
                             grid=r.grids['logit_diff']) for s,r in d.runs.items()})


def table(base, config):
    a = analysis()
    blind = json.loads((source / 'results/phase11_stability.json').read_text())
    payloads = {s:read_result(base, f'ioi.discovery.s{s}') for s in range(10)}
    series, spans = a.effect_series(payloads)
    fixed_series, _ = a.effect_series(payloads, 'fixed')
    theta = a.thetas('ioi')
    heads, zero = a.head_statistics(series, theta)
    fixed, _ = a.head_statistics(fixed_series, theta)
    blind['circuits']['ioi'] = dict(meta=payloads[0]['meta'], theta=theta, series=series,
        heads=heads, heads_position_fixed=fixed, schemes=a.scheme_statistics(series,spans,theta,heads),
        spans=spans, crux=a.crux(series,heads), zero_sd_heads=zero)
    a.CIRCUITS = ('docstring','greater_than','ioi')
    p1,p2 = a.p1_head_level(blind),a.p2_scheme_level(blind)
    for v in p1['tests'].values():
        v['xp_adjusted_p']=min(1.,config['tests']*v['wilcoxon']['p'])
        v['xp_pass']=v['declared_improvement'] and v['xp_adjusted_p']<config['alpha']
    for v in p2['results'].values():
        v['xp_adjusted_p']=min(1.,config['tests']*v['family_wise_p'])
        v['xp_pass']=v['declared_separating'] and v['xp_adjusted_p']<config['alpha']
    return dict(rows=13, blind=blind, p1=p1, p2=p2)


def relationship(base, config):
    a = analysis()
    pub = set(config['sets']['ioi']['published'])
    rows=[]
    for name,heads in config['sets']['ioi'].items():
        if name=='published': continue
        r=read_result(base,f'ioi.{name}.s0')
        rows.append(dict(candidate=name, jaccard=len(pub&set(heads))/len(pub|set(heads)), R=r['R'], verdict=r['verdict']))
    if any(r['R'] is None for r in rows):
        return dict(rows=rows,rho=None,p=None,adjusted_p=None,verdict='INCONCLUSIVE')
    xs,ys=[r['jaccard'] for r in rows],[r['R'] for r in rows]
    rho=a.spearman(xs,ys)
    if not math.isfinite(rho):
        return dict(rows=rows,rho=None,p=None,adjusted_p=None,verdict='INCONCLUSIVE')
    rng=random.Random(4242)
    count=0
    for _ in range(20000):
        shuffled=ys[:]
        rng.shuffle(shuffled)
        count+=a.spearman(xs,shuffled)>=rho-1e-12
    p=(1+count)/20001
    adjusted=min(1.,p*config['tests'])
    return dict(rows=rows,rho=rho,p=p,adjusted_p=adjusted,
                verdict='PASS' if rho>0 and adjusted<config['alpha'] else 'INCONCLUSIVE')


def stability(u, base, config):
    out={}
    for name in config['sets'][u['task']]:
        rows=[read_result(base,f"{u['task']}.{name}.s{s}") for s in range(3)]
        values=[r['R'] for r in rows if r['R'] is not None]
        out[name]=dict(R=values,sd=statistics.stdev(values) if len(values)>1 else None,
            range=max(values)-min(values) if values else None,
            matching_fraction=statistics.fmean(r['verdict']==rows[0]['verdict'] for r in rows[1:]))
    return dict(candidates=out, inferential=False)


def execute(u, config, base):
    if u['kind']=='scrub': return scrub(u,config)
    if u['kind']=='discovery': return discovery(u,config)
    if u['kind']=='errors': return errors(u['task'])
    if u['kind']=='table': return table(base,config)
    if u['kind']=='relationship': return relationship(base,config)
    if u['kind']=='stability': return stability(u,base,config)
    raise ValueError('unknown unit kind')

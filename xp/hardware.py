import csv
import json
import subprocess
import sys


def query(fields, kind='gpu'):
    cmd = ['nvidia-smi', f'--query-{kind}={fields}', '--format=csv,noheader,nounits']
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=15)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or 'nvidia-smi failed')
    return list(csv.reader(p.stdout.strip().splitlines(), skipinitialspace=True))


def inventory():
    rows = query('index,uuid,name,memory.total,driver_version')
    try:
        caps = {r[0]:r[1] for r in query('uuid,compute_cap')}
    except RuntimeError:
        caps = {}
    out = [dict(index=int(r[0]), uuid=r[1], name=r[2], vram_mib=float(r[3]),
                driver=r[4], capability=caps.get(r[1], 'unavailable')) for r in rows]
    if not out:
        raise RuntimeError('no NVIDIA GPUs detected')
    return out


def processes():
    out = {}
    for uuid, pid in query('gpu_uuid,pid', 'compute-apps'):
        try:
            out.setdefault(uuid, set()).add(int(pid))
        except ValueError as e:
            raise RuntimeError('compute PID visibility unavailable, refusing allocation') from e
    return out


def foreign(gpu, jobs, own):
    return jobs.get(gpu['uuid'], set()) - set(own)


def expected(gpus):
    return len(gpus)==4 and all('P100' in g['name'] and 15000<=g['vram_mib']<=17000 for g in gpus)


def torch_info(require_pascal=True):
    import torch
    from importlib.metadata import version
    out = dict(torch=torch.__version__, cuda=torch.version.cuda,
               arch_list=torch.cuda.get_arch_list(), available=torch.cuda.is_available(),
               transformer_lens=version('transformer_lens'), devices=[])
    for i in range(torch.cuda.device_count()):
        prop=torch.cuda.get_device_properties(i)
        out['devices'].append(dict(name=prop.name, vram=prop.total_memory,
                                   capability=f'sm_{prop.major}{prop.minor}'))
    if require_pascal and 'sm_60' not in out['arch_list']:
        raise RuntimeError('installed torch build lacks sm_60: '+json.dumps(out))
    if not out['available'] or not any(d['capability'] in out['arch_list'] for d in out['devices']):
        raise RuntimeError('no compatible CUDA GPU: '+json.dumps(out))
    return out


if __name__=='__main__':
    print(json.dumps(torch_info(), indent=2))

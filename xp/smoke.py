import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import common
import runner
import work
from digest import generate
from hardware import available, compatible, foreign, inventory
from hardware import processes
from sampling import Scrub


def unit(name,device='cpu',deps=None,**kwargs):
    return dict(id=name,device=device,experiment=1,kind='fake',deps=deps or [],core=True,
                priority=int(name.split('.')[-1]),output=f'results/{name}.json',
                cuda=bool(os.environ.get('XP_SMOKE_GPU')) and device=='gpu',**kwargs)


def wait_for(fn,timeout=8):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        result=fn()
        if result: return result
        time.sleep(.02)
    raise AssertionError('timed out waiting for smoke condition')


class Smoke(unittest.TestCase):
    def test_dynamic_gpu_inventory(self):
        for count in (1,2,4,6):
            rows=[[str(i*2),f'GPU-{i}','Tesla P100','16384','test'] for i in range(count)]
            caps=[[f'GPU-{i}','6.0'] for i in range(count)]
            with patch('hardware.query',side_effect=[rows,caps]):
                cards=inventory()
            self.assertEqual(len(cards),count)
            self.assertEqual(len(available(cards,{})),count)
            self.assertEqual(len(available(cards,{'GPU-0':{123}})),count-1)
            self.assertEqual(available(cards,{},selected={0}),[cards[0]])
            with self.assertRaises(RuntimeError): available(cards,{},selected={999})
            info=dict(arch_list=['sm_60'],devices=[dict(capability='sm_60') for _ in cards])
            self.assertEqual(compatible(cards,info),cards)
            info['devices'][0]['capability']='sm_120'
            self.assertEqual(compatible(cards,info),cards[1:])
            info['devices'].pop()
            with self.assertRaises(RuntimeError): compatible(cards,info)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='xp-smoke-')
        self.base=Path(self.tmp.name)
        self.config={'tests':40,'units':[]}
        self.db=None

    def tearDown(self):
        if self.db: self.db.close()
        self.tmp.cleanup()

    def prepare(self,units):
        self.config['units']=units
        self.db=common.initialize(self.base,units,'synthetic')
        common.recover(self.db,self.base,'synthetic')

    def run_queue(self,gpus=None,occupancy=None,hours=None):
        runner.stopping=False
        if gpus is None and os.environ.get('XP_SMOKE_GPU'):
            gpus=[dict(index=0,uuid=os.environ['XP_SMOKE_GPU'])]
        return runner.supervise(self.db,self.base,self.config,'synthetic',None,gpus or [],
                                time.monotonic(),'synthetic',fake=True,occupancy=occupancy,hours=hours)

    def states(self):
        return {r['id']:r['state'] for r in self.db.execute('SELECT * FROM units')}

    def test_success_concurrency_skip_and_checksum(self):
        units=[unit(f'u.{i}','gpu',delay=.03) for i in range(12)]
        self.prepare(units)
        gpus=[dict(index=0,uuid=os.environ['XP_SMOKE_GPU'])] if os.environ.get('XP_SMOKE_GPU') else [dict(index=i,uuid=str(i)) for i in range(2)]
        self.run_queue(gpus)
        self.assertEqual(set(self.states().values()),{'DONE'})
        self.assertTrue(all(r[0]==1 for r in self.db.execute('SELECT attempts FROM units')))
        times={u['id']:(self.base/u['output']).stat().st_mtime_ns for u in units}
        common.recover(self.db,self.base,'synthetic')
        self.run_queue(gpus)
        self.assertEqual(times,{u['id']:(self.base/u['output']).stat().st_mtime_ns for u in units})
        path=self.base/units[0]['output']
        p=json.loads(path.read_text())
        p['data']['value']='corrupted'
        path.write_text(json.dumps(p))
        common.recover(self.db,self.base,'synthetic')
        self.assertEqual(self.states()['u.0'],'PENDING')
        self.run_queue(gpus)
        self.assertTrue(common.valid(path,units[0],'synthetic'))

    def test_retry_failed_dependency_and_independent_work(self):
        device='gpu' if os.environ.get('XP_SMOKE_GPU') else 'cpu'
        self.prepare([unit('u.0',device,failures=1),unit('u.1',device,failures=2),unit('u.2',device,deps=['u.1']),unit('u.3',device)])
        self.run_queue()
        self.assertEqual(self.states(),{'u.0':'DONE','u.1':'FAILED','u.2':'BLOCKED','u.3':'DONE'})
        attempts=dict(self.db.execute('SELECT id,attempts FROM units'))
        self.assertEqual(attempts['u.0'],2)
        self.assertEqual(attempts['u.1'],2)
        text=(self.base/'DIGEST.md').read_text()
        self.assertIn('BLOCKED',text)
        self.assertIn('FAILED',text)

    def test_stop_and_partial_digest_then_resume(self):
        device='gpu' if os.environ.get('XP_SMOKE_GPU') else 'cpu'
        self.prepare([unit(f'u.{i}',device,delay=.25) for i in range(5)])
        def request():
            db=common.connect(self.base)
            wait_for(lambda:db.execute("SELECT 1 FROM units WHERE state='RUNNING'").fetchone())
            (self.base/'STOP').touch()
            db.close()
        thread=threading.Thread(target=request)
        thread.start()
        self.run_queue()
        thread.join()
        self.assertEqual(sum(s=='DONE' for s in self.states().values()),1)
        self.assertIn('NOT YET RUN',(self.base/'DIGEST.md').read_text())
        self.assertIn('PRESENT',(self.base/'DIGEST.md').read_text())
        (self.base/'STOP').unlink()
        common.recover(self.db,self.base,'synthetic')
        self.run_queue()
        self.assertEqual(set(self.states().values()),{'DONE'})

    def test_kill_launcher_parent_death_and_resume(self):
        device='gpu' if os.environ.get('XP_SMOKE_GPU') else 'cpu'
        self.prepare([unit('u.0',device,delay=1),unit('u.1',device)])
        gpus=[dict(index=0,uuid=os.environ['XP_SMOKE_GPU'])] if os.environ.get('XP_SMOKE_GPU') else []
        script=self.base/'launch.py'
        script.write_text('import sys,json,time\nfrom pathlib import Path\n'
            +f'sys.path.insert(0,{str(common.BASE)!r})\nimport common,runner\n'
            +f'base=Path({str(self.base)!r})\n'
            +f'config=json.loads({json.dumps(self.config)!r})\n'
            +f'db=common.connect(base)\nrunner.supervise(db,base,config,"synthetic",None,{gpus!r},time.monotonic(),"synthetic",fake=True)\n')
        proc=subprocess.Popen([sys.executable,str(script)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            row=wait_for(lambda:self.db.execute("SELECT owner FROM units WHERE state='RUNNING'").fetchone())
            owner=row[0]
            if os.environ.get('XP_SMOKE_GPU'):
                wait_for(lambda:int(owner.split(':')[0]) in processes().get(os.environ['XP_SMOKE_GPU'],set()))
            proc.kill()
            proc.wait(timeout=5)
            wait_for(lambda:not common.alive(owner))
            self.assertFalse((self.base/'results/u.0.json').exists())
            common.recover(self.db,self.base,'synthetic')
            self.run_queue()
            self.assertEqual(set(self.states().values()),{'DONE'})
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_atomic_payload_and_lost_claim(self):
        u=unit('u.0')
        self.prepare([u])
        owner=common.identity()
        claimed=common.claim(self.db,'cpu',owner,self.base)
        with patch('common.os.replace',side_effect=OSError('simulated interrupted rename')):
            with self.assertRaises(OSError):
                common.publish(self.db,self.base,claimed,'synthetic',{'value':1},owner)
        self.assertFalse((self.base/u['output']).exists())
        with self.db:
            self.db.execute("UPDATE units SET owner='999999:0:dead'")
        common.recover(self.db,self.base,'synthetic')
        self.assertFalse(list((self.base/'results').glob('*.tmp')))
        claimed=common.claim(self.db,'cpu',owner,self.base)
        common.publish(self.db,self.base,claimed,'synthetic',{'value':1},owner)
        with self.assertRaises(RuntimeError):
            common.publish(self.db,self.base,claimed,'synthetic',{'value':2},owner)
        self.assertEqual(json.loads((self.base/u['output']).read_text())['data']['value'],1)

    def test_foreign_gpu_draining_and_own_processes(self):
        self.prepare([unit('u.0','gpu',delay=.7)])
        gpu=dict(index=0,uuid=os.environ.get('XP_SMOKE_GPU','fake'))
        self.assertFalse(foreign(gpu,{gpu['uuid']:{123}},{123}))
        def occupied(workers):
            row=self.db.execute("SELECT 1 FROM units WHERE state='RUNNING'").fetchone()
            own={w['proc'].pid for w in workers.values()}
            return {gpu['uuid']:own|({999999} if row else set())}
        workers=self.run_queue([gpu],occupied)
        self.assertEqual(workers['gpu0']['state'],'RELEASED')
        self.assertEqual(self.states()['u.0'],'PENDING')
        self.assertFalse((self.base/'results/u.0.json').exists())
        common.recover(self.db,self.base,'synthetic')
        self.run_queue([gpu])
        self.assertEqual(self.states()['u.0'],'DONE')

    def test_time_budget(self):
        device='gpu' if os.environ.get('XP_SMOKE_GPU') else 'cpu'
        self.prepare([unit(f'u.{i}',device,delay=.3) for i in range(4)])
        self.run_queue(hours=.00012)
        self.assertIn('PENDING',self.states().values())
        self.assertNotIn('RUNNING',self.states().values())

    def test_plan_untracked_uncommitted_modified_refusals(self):
        root=self.base
        (root/'xp').mkdir()
        (root/'scripts').mkdir()
        for name in ['xp/PLAN.md','xp/run.sh','xp/setup.sh','xp/constraints.txt','scripts/check_env.py']:
            (root/name).write_text('test\n')
        subprocess.run(['git','init','-q',str(root)],check=True)
        for key in ('user.name','user.email'):
            value=common.git('config','--get',key)
            subprocess.run(['git','-C',str(root),'config',key,value],check=True)
        with patch('common.ROOT',root),patch('common.BASE',root/'xp'):
            with self.assertRaises(RuntimeError): common.preregistered()
            subprocess.run(['git','-C',str(root),'add','.'],check=True)
            with self.assertRaises(RuntimeError): common.preregistered()
            message=root/'message'
            message.write_text('xp smoke fixture\n')
            subprocess.run(['git','-C',str(root),'commit','-q','-F',str(message)],check=True)
            self.assertEqual(len(common.preregistered()),64)
            (root/'xp/PLAN.md').write_text('changed\n')
            with self.assertRaises(RuntimeError): common.preregistered()

    def test_cpu_guard_and_correlated_error_math(self):
        code='import sys\nsys.path.insert(0,'+repr(str(common.BASE))+')\nfrom worker import guard\nguard()\nimport torch\n'
        p=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('prohibited model import',p.stderr)
        results=self.base/'results'
        results.mkdir()
        fixture=dict(effects={'a':{'0.0':.1,'0.1':.1,'0.2':0,'0.3':0},
                              'b':{'0.0':.1,'0.1':.1,'0.2':0,'0.3':0}},
                     floors={'a':{'threshold':.02},'b':{'threshold':.02}},
                     scored_before={'per_scheme':{'a':{'matches':['0.0'],'misses':['0.2']}}})
        (results/'phase9_toy.json').write_text(json.dumps(fixture))
        with patch('work.source',self.base):
            data=work.errors('toy')['variants']['own_theta']
        self.assertEqual(data['rho'],1)
        self.assertEqual(data['n_eff'],1)
        self.assertEqual(data['observed_majority'],.5)
        self.assertGreaterEqual(data['cantelli'],.5)

    def test_recursive_semantic_sampling_and_shared_unimportant_donor(self):
        parents={'sum':['a','b','c'],'a':['input'],'b':['input'],'c':['input'],'input':[]}
        functions={'sum':lambda xs:xs,'input':lambda x:x,'a':lambda xs:xs[0],
                   'b':lambda xs:xs[0], 'c':lambda xs:xs[0]}
        s=Scrub(parents,functions,{'sum':['a'],'a':['input']},
                {'a':lambda x:x%2,'input':lambda x:x},range(10))
        samples=[s.run('sum',0,random.Random(i)) for i in range(100)]
        self.assertTrue(all(a%2==0 and b==c for a,b,c in samples))
        self.assertGreater(len({a for a,b,c in samples}),1)
        self.assertGreater(len({b for a,b,c in samples}),1)

    def test_gate_paths_simulated_and_no_claim_on_failure(self):
        self.config['units']=[unit('u.0')]
        gpu=dict(index=0,uuid='fake',name='fake',vram_mib=16384,driver='simulated')
        for passed in (False,True):
            runner.stopping=False
            def spawn(*args,**kwargs):
                common.write_json(self.base/'gate.json',dict(reproduction=dict(passed=passed),environment={},probe_seconds=.01))
                proc=Mock()
                proc.poll.return_value=0
                proc.returncode=0
                return dict(proc=proc,owner=None,name='gate',gpu=gpu,state='STOPPED')
            with patch.multiple(runner,BASE=self.base), \
                 patch('runner.plan',return_value=self.config), \
                 patch('runner.preregistered',return_value='synthetic'), \
                 patch('runner.git',return_value='synthetic'), \
                 patch('runner.inventory',return_value=[gpu]), \
                 patch('runner.processes',return_value={}), \
                 patch('runner.snapshot',return_value=self.base), \
                 patch('runner.spawn',side_effect=spawn), \
                 patch('runner.generate',side_effect=lambda:generate(self.base,self.config)), \
                 patch('runner.supervise') as supervised, \
                 patch('runner.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(dict(devices=[dict(capability='sm_60')],arch_list=['sm_60'])))), \
                 patch.object(sys,'argv',['runner.py','--yes']):
                code=runner.main()
                self.assertEqual(code,0 if passed else 2)
                self.assertEqual(supervised.called,passed)
            db=common.connect(self.base)
            self.assertEqual(db.execute('SELECT state FROM units').fetchone()[0],'PENDING')
            db.close()

    def test_preservation_statistics(self):
        try: import scipy
        except ImportError: self.skipTest('scipy unavailable')
        cfg=dict(preservation=.8,tests=40,alpha=.05)
        clean=[1.]*128
        floor=[0.]*128
        self.assertEqual(work.summary(clean,floor,[.9]*128,cfg)['verdict'],'PASS')
        self.assertEqual(work.summary(clean,floor,[.5]*128,cfg)['verdict'],'FAILURE')
        self.assertEqual(work.summary(clean,clean,[1.]*128,cfg)['verdict'],'INCONCLUSIVE')

    def test_tiny_random_transformer_scrub(self):
        try:
            import torch
            from transformer_lens import HookedTransformer,HookedTransformerConfig
        except ImportError:
            self.skipTest('torch/TransformerLens unavailable')
        torch.set_num_threads(1)
        model=HookedTransformer(HookedTransformerConfig(n_layers=2,d_model=16,n_ctx=8,
            d_head=4,n_heads=4,d_vocab=32,act_fn='relu',d_mlp=32,device='cpu',seed=42))
        model.tokenizer=SimpleNamespace(eos_token_id=0,pad_token_id=0,bos_token_id=0,padding_side='right')
        gen=torch.Generator().manual_seed(17)
        tokens=[torch.randint(1,32,(5+i%3,),generator=gen) for i in range(256)]
        ds=SimpleNamespace(io_token_ids=torch.full((128,),2),s_token_ids=torch.full((128,),3))
        task=SimpleNamespace(dataset=lambda *args,**kwargs:ds)
        cfg=dict(prompts=128,draws=2,preservation=.8,tests=40,alpha=.05)
        u=dict(task='ioi',heads=['0.0','1.1'],seed=0,candidate='synthetic')
        work.pools.clear()
        with torch.no_grad(),patch('work.load_model',return_value=model), \
             patch('work.task_spec',return_value=task),patch('work.variants',return_value=tokens):
            result=work.scrub(u,cfg)
            repeat=work.scrub(u,cfg)
        self.assertLess(result['reconstruction_error'],.001)
        self.assertEqual(result,repeat)
        self.assertEqual(len(result['scrub']),128)
        self.assertNotEqual(result['clean'],result['scrub'])
        work.pools.clear()


if __name__=='__main__':
    unittest.main(verbosity=2)

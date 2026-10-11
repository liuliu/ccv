"""Locate the causal floating-PV head-width and sequence-length crossovers.

Both variants use quantizePV=false, identical inputs, rotating samples,
and full preparation. Baseline uses the original tile policy; candidate uses
blockC=32 and four SIMD groups, keeping blockD/output tiles unchanged.
Raw per-case results belong outside the repository.
"""
import argparse,json,os,re,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--probe', required=True)
p.add_argument('--device')
p.add_argument('--bundle', default='com.liu.NAQuantizePVCrossover')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sweep', choices=['head', 'rc'], default='head')
p.add_argument('--rows', default='64,128,256,512,768,1024,1536,2048')
p.add_argument('--columns', default='256,512,1024,2048,4096,8192')
p.add_argument('--widths', default='256')
p.add_argument('--heads', default='8')
p.add_argument('--kv-heads', type=int, help='Keep KV heads fixed while varying query heads')
p.add_argument('--batches', default='1')
p.add_argument('--tails', default='0,1', help='Offsets added to both R and C')
p.add_argument('--precision', type=int, choices=[0,1,2], default=0)
p.add_argument('--sinks', type=int, choices=[0,1,2,3], default=0)
p.add_argument('--hadamard', type=int, choices=[0,1], default=0)
p.add_argument('--samples', type=int, default=9)
p.add_argument('--dynamic', type=int, choices=[0,1,2,3], default=3)
a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=True)
cases=[]
def add(name,d,*,r=1025,c=4097,h=8,b=1,precision=0,mask=0,sinks=0,varlen=0,hadamard=0):
 bd=64 if d>=192 else 32;ot=128 if d>192 else d
 base=[64,bd,8 if d>192 else 4,ot]
 candidate=[32,bd,4,ot]
 cases.append(dict(name=name,values=[r,c,b,h,a.kv_heads or max(1,h//4),precision,1,a.dynamic,4,varlen,d,hadamard,a.samples,0,0,0,1],base=base,candidate=candidate,mask=mask,sinks=sinks))
if a.sweep == 'rc':
 for d in map(int,a.widths.split(',')):
  for b in map(int,a.batches.split(',')):
   for h in map(int,a.heads.split(',')):
    for r in map(int,a.rows.split(',')):
     for c in map(int,a.columns.split(',')):
      if r>c:continue
      for tail in map(int,a.tails.split(',')):
       add(f'd{d}-r{r+tail}-c{c+tail}-b{b}-h{h}-k{a.kv_heads or max(1,h//4)}-p{a.precision}-s{a.sinks}-q{a.hadamard}',d,
           r=r+tail,c=c+tail,b=b,h=h,precision=a.precision,sinks=a.sinks,hadamard=a.hadamard)
else:
 for d in [128,160,176,184,192,200,208,216,224,232,240,248,256,264,272,288,320,384,512]:
  for tail in [0,1]:
   add(f'd{d}-tail{tail}',d,r=1024+tail,c=4096+tail)
 for d in [192,200,256,320]:
  for r,c in [(1,8193),(8,8193),(64,1025),(129,4097),(513,2049)]:
   add(f'd{d}-r{r}-c{c}',d,r=r,c=c)
 for d in [192,200,256]:
  for label,mask,sinks,varlen,hadamard in [('mask',2,0,0,0),('sink',0,3,0,0),('varlen',0,0,1,0),('hadamard',0,0,0,1)]:
   add(f'd{d}-{label}',d,r=513,c=2049,b=2,mask=mask,sinks=sinks,varlen=varlen,hadamard=hadamard)
 for d in [184,192,200,224,256,320]:
  for precision in [1,2]:add(f'd{d}-p{precision}',d,precision=precision)
 for d in [200,256]:
  for precision in [0,1,2]:
   add(f'd{d}-p{precision}-combined',d,r=513,c=2049,b=2,precision=precision,mask=2,sinks=3,varlen=1,hadamard=1)
   if precision:
    for label,mask,sinks,varlen,hadamard in [('mask',2,0,0,0),('sink',0,3,0,0),('varlen',0,0,1,0),('hadamard',0,0,0,1)]:
     add(f'd{d}-p{precision}-{label}',d,r=513,c=2049,b=2,precision=precision,mask=mask,sinks=sinks,varlen=varlen,hadamard=hadamard)
for i,x in enumerate(cases):
 path=out/(x['name']+'.json')
 if path.exists():
  previous=json.loads(path.read_text())
  if previous.get('correct') and all(previous.get(k)==x[k] for k in ['values','base','candidate','mask','sinks']):continue
 env=dict(os.environ,CCV_PV_COMPARE_CONFIGS='1',CCV_PV_CONFIG_0=','.join(map(str,x['base'])),CCV_PV_CONFIG_1=','.join(map(str,x['candidate'])),CCV_PV_MASK=str(x['mask']),CCV_PV_SINKS=str(x['sinks']))
 cmd=[a.probe,*map(str,x['values'])]
 if a.device:
  cmd=['xcrun','devicectl','device','process','launch','--device',a.device,'--terminate-existing','--console',a.bundle,*map(str,x['values'])]
  for k,v in list(env.items()):
   if k.startswith('CCV_PV_'):env['DEVICECTL_CHILD_'+k]=v
 start=time.monotonic();log=path.with_suffix('.log')
 with log.open('w') as f:
  try:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=100).returncode
  except subprocess.TimeoutExpired:code='timeout'
 text=log.read_text();m=re.search(r'config_timing baseline_ms=(\S+) candidate_ms=(\S+)',text);times=list(map(float,m.groups())) if m else None
 record=dict(**x,exit=code,correct=code==0 and times is not None and bool(re.search(r'^correct=1$',text,re.M)),times=times,speedup=times[0]/times[1] if times else None,thermal=re.findall(r'thermal_(?:state|end)=(\d+)',text),seconds=time.monotonic()-start)
 path.write_text(json.dumps(record,indent=2));print(f'{i+1}/{len(cases)}',x['name'],record['correct'],round(record['speedup'],3) if times else code,flush=True)
 if not record['correct']:raise SystemExit('Inspect failure: '+str(log))

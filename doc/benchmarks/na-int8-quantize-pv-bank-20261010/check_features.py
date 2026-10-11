"""Exercise odd lengths, masks, sinks, causal attention and packed sequences."""
import argparse,json,os,re,subprocess,time
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--probe',required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--filter',default='')
parser.add_argument('--pair-only',action='store_true')
parser.add_argument('--device')
parser.add_argument('--bundle',default='com.liu.NAQuantizePVFeatures')
parser.add_argument('--timeout',type=int,default=180)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
# mask mode, sink mode, causal, varlen, Hadamard, R, C
features=[
 ('mask-shared',1,0,0,0,0,63,65),
 ('mask-batched-sinks',2,1,0,0,0,129,257),
 ('causal-mask',2,0,1,0,0,65,63),
 ('causal-mask-sinks',2,3,1,0,0,63,129),
 ('fully-masked',3,0,0,0,0,64,65),
 ('fully-masked-sinks',3,1,1,0,0,65,64),
 ('varlen-sinks',0,1,0,1,0,65,129),
 ('varlen-causal-sinks',0,3,1,1,0,129,65),
 ('varlen-mask-sinks',1,2,1,1,0,65,129),
 ('batched-varlen-mask',2,1,0,1,0,65,129),
 ('hadamard-mask-sinks',2,3,1,0,1,65,129),
 ('hadamard-varlen-sinks',0,1,1,1,1,65,129),
 ('hadamard-padded-sinks',0,1,0,0,1,65,1057),
 ('hadamard-causal-padded-sinks',0,3,1,0,1,65,1057),
 ('decode-sinks',0,3,1,0,0,1,1057),
 ('decode-mask-sinks',2,1,1,0,0,8,129),
 ('decode-varlen-sinks',1,3,0,1,0,8,129),
 ('decode-hadamard-sinks',0,1,1,0,1,1,1057),
 ('decode-tail-r1',0,0,0,0,0,1,1057),
 ('decode-tail-r8',0,0,0,0,0,8,1057),
 ('decode-causal-tail-r1',0,0,1,0,0,1,1057),
 ('decode-causal-tail-r8',0,0,1,0,0,8,1057),
 ('batched-varlen-mask-no-sinks',2,0,0,1,0,65,129),
]
cases=[]
for precision in range(3):
 for d in [64,128,256]:
  for i,(name,mask,sinks,causal,varlen,hadamard,r,c) in enumerate(features):
   cases.append((f'{name}-p{precision}-d{d}',[r,c,2,4,2,precision,causal,3 if i%2 else 0,0,varlen,d,hadamard,3,0,0,0,1],mask,sinks))
for causal in [0,1]:
 for c in [1023,1024,1025]:
  for mask,sinks in [(0,1),(2,3)]:
   cases.append((f'tail-perf-causal{causal}-c{c}-mask{mask}',[513,c,2,4,2,0,causal,3,4,0,128,0,3,0,0,0,1],mask,sinks))
   cases.append((f'tail-perf-causal{causal}-c{c}-mask{mask}-varlen1',[513,c,2,4,2,0,causal,3,4,1,128,0,3,0,0,0,1],mask,sinks))
for precision in range(3):
 for d in [8,80,192,224]:
  for varlen in [0,1]:
   cases.append((f'head-tail-p{precision}-d{d}-varlen{varlen}',[65,129,2,4,2,precision,1,3,0,varlen,d,0,3,0,0,0,1],2,3))
for name,values,mask,sinks in cases:
 if not re.search(args.filter,name):continue
 dest=args.output/(name+'.json')
 if dest.exists():continue
 cmd=[args.probe,*map(str,values)]
 env=dict(os.environ,CCV_PV_MASK=str(mask),CCV_PV_SINKS=str(sinks))
 if args.pair_only:env['CCV_PV_PAIR_ONLY']='1'
 if args.device:
  cmd=['xcrun','devicectl','device','process','launch','--device',args.device,'--terminate-existing','--console',args.bundle,*map(str,values)]
  for key in ['CCV_PV_MASK','CCV_PV_SINKS','CCV_PV_PAIR_ONLY']:
   if key in env:env['DEVICECTL_CHILD_'+key]=env[key]
 log=dest.with_suffix('.log');start=time.time()
 with log.open('w') as output:
  try:code=subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=args.timeout).returncode
  except subprocess.TimeoutExpired:code='timeout'
 text=log.read_text()
 checks=[dict(variant=int(v),full_reference=int(f),rel_l2=float(r),max_abs=float(m),correct=bool(int(ok))) for v,f,r,m,ok in re.findall(r'attention_check variant=(\d+) full_reference=(\d+) rel_l2=(\S+) max_abs=(\S+) correct=(\d)',text)]
 timing=re.search(r'timing full=1 int8_ms=(\S+) hybrid_ms=(\S+)' + ('' if args.pair_only else r' na_ms=(\S+)'),text)
 record=dict(command=cmd,features=dict(mask=mask,sinks=sinks,pair_only=args.pair_only),exit=code,seconds=time.time()-start,correct=bool(re.search(r'^correct=1$',text,re.M)),false_correct=any(c['variant']==1 and c['correct'] for c in checks),checks=checks,thermal=re.findall(r'thermal_(?:state|end)=(\d+)',text),
             timing=list(map(float,timing.groups())) if timing else None)
 dest.write_text(json.dumps(record,indent=2));print(name,'false_correct='+str(record['false_correct']),code,[round(c['rel_l2'],5) for c in checks],flush=True)

import subprocess,time,os,json,re,sys,hashlib
from pathlib import Path
root=Path('/Users/liu/workspace/ccv');out=Path('/tmp/ccv-int8-tuning');sys.path.insert(0,str(root/'bin/mfa'))
from na_benchmark_manifest import manifest
ids=['flux2-512-attention','qwen21-1024-attention','attn-d256-r4096-c4096','attn-partition-r4096-c24576-h48','attn-d128-r4097-c4097','attn-dynamic-d256']
byid={n:c for c in manifest()['workloads'] for n in [c['id']]+c['aliases']}
selected=[byid[x] for x in ids]
bins={'base':Path('/tmp/ccv-na-validation/base-probe'),'branch':Path('/tmp/ccv-na-validation/branch-probe'),'native':out/'native-probe'}
label=sys.argv[1] if len(sys.argv)>1 else 'attention-final'; dest=out/label;dest.mkdir(exist_ok=True)
(dest/'metadata.json').write_text(json.dumps(dict(shapes=selected,sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in bins.items()},warmup=.5,cooldown=8,order=['base','native','native','base']),indent=2))
env=dict(os.environ,CCV_NA_WARMUP_SECONDS='0.5')
print('Cooling 30 seconds',flush=True)
time.sleep(30)
def metric(s,k):
 m=re.search(r'(?<![\w])'+k+r'=([^\s]+)',s);return m.group(1) if m else None
for c in selected:
 for i,arm in enumerate(['base','native','native','base']):
  time.sleep(8)
  before=subprocess.check_output(['/tmp/ccv-na-validation/thermal'],text=True).strip()
  while before!='0':
   print('Cooling thermal',before,flush=True);time.sleep(30);before=subprocess.check_output(['/tmp/ccv-na-validation/thermal'],text=True).strip()
  cmd=[str(bins[arm]),c['operation'],*map(str,c['shape']),'3','21',str(int(c['causal'])),'1','0',str(c['dispatch_flags'])]
  r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
  tag=f'{c["id"]}-{i}-{arm}';(dest/(tag+'.log')).write_text(r.stdout)
  d=dict(id=c['id'],shape=c['shape'],arm=arm,order=i,rc=r.returncode,median_ms=metric(r.stdout,'gpu_median_ms'),l2=metric(r.stdout,'normalized_l2'),thermal=before,thermal_end=subprocess.check_output(['/tmp/ccv-na-validation/thermal'],text=True).strip(),cores=metric(r.stdout,'gpu_core_count'),command=cmd)
  with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
  print(d,flush=True)
  if r.returncode or d['median_ms'] is None:raise SystemExit(1)
print('DONE',flush=True)

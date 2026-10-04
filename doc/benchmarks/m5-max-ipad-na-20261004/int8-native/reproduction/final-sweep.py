import subprocess,time,os,json,re,sys,hashlib
from pathlib import Path
root=Path('/Users/liu/workspace/ccv');out=Path('/tmp/ccv-int8-tuning');sys.path.insert(0,str(root/'bin/mfa'))
from na_benchmark_manifest import manifest
ids=['flux2-1024-projection','flux2-1024-ffn-up','flux2-1024-ffn-down','qwen21-1024-projection','gemm-rows-512-6144-6144','gemm-rows-2048-6144-6144','gemm-rows-4097-6144-6144','gemm-aspect-4096-6144-16384','gemm-aspect-2048-2048-8192','gemm-rows-8192-1536-4096','gemm-edges-n4097-k8193','gemm-edges-n1536-k32768','local-qwen35-4b-m1-up','local-qwen35-4b-m128-up','local-qwen35-4b-m2048-up','local-qwen35-27b-m2048-down']
byid={n:c for c in manifest()['workloads'] for n in [c['id']]+c['aliases']}
selected=[byid[x] for x in ids]
bins={'base':Path('/tmp/ccv-na-validation/base-probe'),'branch':Path('/tmp/ccv-na-validation/branch-probe'),'native':out/'native-probe'}
label=sys.argv[1] if len(sys.argv)>1 else 'mac-final'; dest=out/label;dest.mkdir(exist_ok=True)
(dest/'metadata.json').write_text(json.dumps(dict(shapes=selected,sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in bins.items()},warmup=.5,cooldown=8,order=['base','branch','native','native','branch','base']),indent=2))
env=dict(os.environ,CCV_NA_WARMUP_SECONDS='0.5')
print('Cooling 120 seconds',flush=True)
for _ in range(4):time.sleep(30)
def metric(s,k):
 m=re.search(r'(?<![\w])'+k+r'=([^\s]+)',s);return m.group(1) if m else None
for c in selected:
 for i,arm in enumerate(['base','branch','native','native','branch','base']):
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

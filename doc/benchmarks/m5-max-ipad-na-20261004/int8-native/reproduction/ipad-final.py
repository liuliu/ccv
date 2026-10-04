import subprocess,time,os,json,re,sys,hashlib
from pathlib import Path
root=Path('/Users/liu/workspace/ccv');out=Path('/tmp/ccv-int8-tuning');sys.path.insert(0,str(root/'bin/mfa'))
from na_benchmark_manifest import manifest
ids=['gemm-rows-512-6144-6144','gemm-rows-2048-6144-6144','flux2-1024-projection','flux2-1024-ffn-down','gemm-edges-n4097-k8193','attn-d256-r4096-c4096','flux2-512-attention','qwen21-1024-attention','attn-partition-r4096-c24576-h48','attn-d128-r4097-c4097','attn-dynamic-d256']
byid={n:c for c in manifest()['workloads'] for n in [c['id']]+c['aliases']}; selected=[byid[x] for x in ids]
dest=out/'ipad-final';dest.mkdir(exist_ok=True)
apps={k:Path('/tmp/ccv-na-validation')/('xcode-'+k)/'Build/Products/Release-iphoneos/NAInt8TuningApp.app/NAInt8TuningApp' for k in ['base','native']}
(dest/'metadata.json').write_text(json.dumps(dict(shapes=selected,sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in apps.items()},warmup=.5,samples=21,cooldown=20,large_case_cooldown=45,order=['base','native','native','base']),indent=2))
def metric(s,k):
 m=re.search(r'(?<![\w])'+k+r'=([^\s]+)',s);return m.group(1) if m else None
print('Initial cooling 120 seconds',flush=True)
for _ in range(4):time.sleep(30)
for c in selected:
 for i,arm in enumerate(['base','native','native','base']):
  time.sleep(45 if c['id'] in ['flux2-1024-ffn-down','gemm-edges-n4097-k8193','attn-partition-r4096-c24576-h48'] else 20)
  cmd=['xcrun','devicectl','device','process','launch','--device','00008142-001868211E47801C','--terminate-existing','--console','com.liu.NAInferenceValidation.'+arm,c['operation'],*map(str,c['shape']),'3','21',str(int(c['causal'])),'1','0',str(c['dispatch_flags'])]
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=240)
  tag=f'{c["id"]}-{i}-{arm}';(dest/(tag+'.log')).write_text(r.stdout)
  d=dict(id=c['id'],shape=c['shape'],arm=arm,order=i,rc=r.returncode,probe_exit=metric(r.stdout,'probe_exit'),median_ms=metric(r.stdout,'gpu_median_ms'),l2=metric(r.stdout,'normalized_l2'),thermal=metric(r.stdout,'thermal_state'),thermal_end=metric(r.stdout,'thermal_end'),cores=metric(r.stdout,'gpu_core_count'),command=cmd)
  with (dest/'records.jsonl').open('a') as f:f.write(json.dumps(d)+'\n')
  print(d,flush=True)
  if r.returncode or d['probe_exit']!='0' or d['median_ms'] is None or (arm=='native' and d['cores']!='10'):raise SystemExit(1)
  if d['thermal']!='0' or d['thermal_end']!='0':raise SystemExit('Non-nominal thermal state; stop for cooling before accepting timings.')
print('DONE',flush=True)

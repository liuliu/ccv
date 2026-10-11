import argparse,json,os,re,subprocess
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--device',required=True)
parser.add_argument('--bundle',default='com.liu.NAQuantizePVFeatures')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
for precision in range(3):
 for r,c in [(1,1056),(8,2080),(128,1056)]:
  name=f'p{precision}-r{r}-c{c}';values=[r,c,2,4,2,precision,1,3,0,0,256,0,3,0,0,0,1]
  cmd=['xcrun','devicectl','device','process','launch','--device',args.device,'--terminate-existing','--console',args.bundle,*map(str,values)]
  env=dict(os.environ)
  env['DEVICECTL_CHILD_CCV_PV_VALIDATE_QUANTIZERS']='1'
  for key in ('MTL_DEBUG_LAYER','MTL_SHADER_VALIDATION','MTL_SHADER_VALIDATION_REPORT_TO_STDERR','MTL_SHADER_VALIDATION_ABORT_ON_FAULT'):env['DEVICECTL_CHILD_'+key]='1'
  log=out/(name+'.log')
  with log.open('w') as f:code=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
  text=log.read_text();correct=code==0 and 'quantizer_validation=1' in text and 'probe_exit=0' in text
  (out/(name+'.json')).write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('DEVICECTL_CHILD_')},exit=code,correct=correct),indent=2));print(name,correct,flush=True)
  if not correct:raise SystemExit(1)
  with (out/(name+'-reference.log')).open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
  text=(out/(name+'-reference.log')).read_text();correct=code==0 and 'correct=1' in text and 'probe_exit=0' in text
  (out/(name+'-reference.json')).write_text(json.dumps(dict(command=cmd,exit=code,correct=correct),indent=2));print(name+'-reference',correct,flush=True)
  if not correct:raise SystemExit(1)

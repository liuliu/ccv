"""Run the full 128-case attention bank, preserving raw three-way GPU timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from na_benchmark_manifest import manifest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--probe', default='/private/tmp/ccv-quantize-pv-mac/probe')
parser.add_argument('--device')
parser.add_argument('--debug-long', action='store_true')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--samples', type=int, default=15)
parser.add_argument('--filter', default='')
parser.add_argument('--precision', type=int, default=0)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--long-profile', action='store_true')
parser.add_argument('--pair-only', action='store_true', help='Compare only true/false when the independent NAAttention baseline cannot run')
parser.add_argument('--bundle', default='com.liu.NAFloatPVAttentionExperiment')
args = parser.parse_args()
if args.pair_only:
    os.environ['DEVICECTL_CHILD_CCV_PV_PAIR_ONLY' if args.device else 'CCV_PV_PAIR_ONLY']='1'
args.output.mkdir(parents=True, exist_ok=True)
cases = [c for c in manifest()['workloads'] if c['operation']=='attention' and re.search(args.filter, c['id'])]
# Long H3 cases go last so resource failures cannot hide normal bank coverage.
cases.sort(key=lambda c: 'long' in c['tags'])
metadata = dict(bank_source='be445250:bin/mfa/na_benchmark_manifest.py',
    base_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
    cases=cases, arguments=vars(args) | {'output':str(args.output)},
    variants=['quantizePV=true (default/tip behavior)', 'quantizePV=false'] + ([] if args.pair_only else ['NAAttention']),
    timing=('GPU timestamps, full forward only; rotating variant order; >=1 warmup round and >=300ms aggregate GPU work; capped at 3 samples' if args.long_profile else 'GPU timestamps, preparation included and kernel only; rotating variant order; >=5 warmup rounds and >=300ms aggregate GPU work'),
    scope='NAInt8Attention descriptor-level forward bank; direct kernels including decode, independent of production frontend fallback policies')
if not args.device: metadata['binary_sha256']=hashlib.sha256(Path(args.probe).read_bytes()).hexdigest()
(args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
for repeat in range(args.repeat):
 for i,c in enumerate(cases):
    case_id = c['id'] + (f'-repeat{repeat}' if args.repeat > 1 else '')
    result_file=args.output/(case_id+'.json')
    if result_file.exists(): continue
    r,k,d,b,hq,hk=c['shape']
    values=[r,k,b,hq,hk,args.precision,int(c['causal']),c['dispatch_flags'],4,0,d,0,args.samples]
    if args.long_profile: values += [0,0,0,1]
    cmd=[args.probe,*map(str,values)]
    if args.device:
        cmd=['xcrun','devicectl','device','process','launch','--device',args.device,'--terminate-existing','--console',args.bundle,*map(str,values)]
    # A short idle interval between workloads limits accumulated device heat.
    time.sleep(3 if args.device else 1)
    start=time.time()
    log=args.output/(case_id+'.log')
    with log.open('w') as f:
        try:
            if args.device and args.debug_long and (r*k*hq*d > 100000000000 or 'long' in c['tags']):
                from launch_ipad import run
                status=run(args.device,values,log,bundle=args.bundle)
            else:
                p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
                status=p.returncode
        except subprocess.TimeoutExpired: status='timeout'
    output=log.read_text()
    record=dict(case=c,command=cmd,exit=status,seconds=time.time()-start,
        correct=bool(re.search(r'^correct=1$', output, re.M)) and (not args.device or 'probe_exit=0' in output),
        thermal=re.findall(r'thermal_(?:state|end)=(\d+)',output),
        timings=[dict(full=int(full),int8_ms=float(a),hybrid_ms=float(b),na_ms=float(n))
            for full,a,b,n in re.findall(r'timing full=(\d) int8_ms=(\S+) hybrid_ms=(\S+) na_ms=(\S+)',output)])
    if args.pair_only:
        record['timings']=[dict(full=int(full),int8_ms=float(a),hybrid_ms=float(b),na_ms=None)
            for full,a,b in re.findall(r'pair_timing full=(\d) int8_ms=(\S+) hybrid_ms=(\S+) samples=',output)]
    result_file.write_text(json.dumps(record,indent=2))
    print(f'{i+1}/{len(cases)} {case_id} correct={record["correct"]} exit={status} seconds={record["seconds"]:.1f} thermal={record["thermal"]}',flush=True)

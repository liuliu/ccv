#!/usr/bin/env python3
"""Serial shape sweeps with raw commands, validation and GPU timestamps.

Build the four NA probes and mlx_na_bench first. Pass --mlx for comparisons.
Repeated passes reverse order to expose clock/order effects. Failed validation
is recorded and never silently treated as a usable performance measurement.
"""
import argparse
import csv
import os
import math
import struct
import pathlib
import re
import subprocess


def matmul_shapes():
    shapes = [(m,n,k) for m in (64,256,1024,4096) for n in (256,1024,4096)
              for k in (512,2048,4096,8192)]
    shapes += [(m,n,k) for m,n in ((16,8192),(1024,8192),(8192,1024),(8192,4096))
               for k in (2048,4096,16384)]
    shapes += [(1001,2053,1031),(4097,1025,4097),(63,129,511),(127,4096,3072),
               (4096,127,4096),(2048,2048,16384),(4096,4096,6144)]
    return shapes


def attention_shapes():
    shapes = [(s,s,d,1,8,8) for d in (64,80,96,128,192,256) for s in (256,1024,4096)]
    shapes += [(r,c,d,1,8,8) for d in (64,128,256) for r,c in ((128,4096),(4096,128),(1001,2053))]
    shapes += [(1024,1024,d,b,hq,hk) for d in (64,128,256)
               for b,hq,hk in ((4,8,8),(1,24,4),(4,16,4))]
    shapes += [(8192,8192,128,1,8,8)]
    return shapes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite', choices=['fp16','int8','attention','backward','mlx-matmul','tune-backward','tune-attention','tune-na-forward','int8-long','mlx-attention','backward-int8','backward-supported'])
    p.add_argument('output', type=pathlib.Path)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--iterations', type=int, default=25)
    p.add_argument('--warmup-seconds', type=float, default=0.1)
    p.add_argument('--shape', action='append', help='Only run this x-separated shape (repeatable)')
    p.add_argument('--mlx', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent / 'mlx_na_bench')
    args = p.parse_args()
    if min(args.repeats,args.warmup,args.iterations) < 1: p.error('counts must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    bindir = pathlib.Path(__file__).resolve().parent
    jobs = []
    def add(shape,variant,binary,*params):
        jobs.append((shape,variant,[str(bindir/binary),*map(str,params)]))
    w,t = args.warmup,args.iterations
    if args.suite == 'fp16':
        shapes = [(m,n,k,1,0,1) for m,n,k in matmul_shapes()]
        shapes += [(m,n,k,b,ta,tb) for m,n,k in ((256,4096,4096),(4096,1024,4096),(1001,2053,1031))
                   for b,ta,tb in ((1,0,0),(1,1,0),(1,1,1),(4,0,1))]
        for shape in shapes:
            m,n,k,b,ta,tb = shape
            split = 1 if n%64 else (k//3072//2*2 if k>12288 else 4 if k>=8192 else 2 if k>=4096 else 1)
            variants = [('old',(128,64,64),split),('wide',(64,128,512),1),
                        ('tall',(128,64,512),1),('wide-split',(64,128,512),split)]
            for name,tile,sk in variants:
                if name == 'wide-split' and split == 1: continue
                add(shape,name,'na_gemm_splitk_bench',m,n,k,w,t,sk,1,-1,*tile,b,ta,tb)
    elif args.suite in ('int8','int8-long'):
        shapes = matmul_shapes() if args.suite == 'int8' else [
            (4096,4096,12288),(4096,4096,16384),(4096,4096,32768),
            (8192,4096,12288),(4096,8192,16384),(4096,1024,32768),
            (2048,8192,16384),(8192,4096,32768)]
        for shape in shapes:
            for name,tile in [('old',(128,128,128,8)),('small-tile',(64,128,128,4))]:
                add(shape,name,'na_int8_matmul_bench',*shape,w,t,*tile,256,4294967295,4294967295,1,1)
    elif args.suite == 'mlx-matmul':
        for shape in matmul_shapes():
            jobs.append((shape,'mlx',[str(args.mlx),'matmul',*map(str,shape), '1','8','8',str(w),str(t),'0']))
    elif args.suite == 'mlx-attention':
        for shape in attention_shapes():
            jobs.append((shape,'mlx',[str(args.mlx),'attention',*map(str,shape),str(w),str(t),'0']))
    elif args.suite == 'tune-attention':
        for shape in attention_shapes():
            if shape[2] not in (192,256): continue
            for name,tile in [('old',(0,0,0,0)),('C128-D32',(4,128,16,32)),('C64-D32',(4,64,16,32))]:
                add(shape,name,'na_int8_attention_bench',*shape,w,t,16,*tile)
    elif args.suite in ('attention','tune-na-forward'):
        for shape in attention_shapes():
            if args.suite == 'tune-na-forward' and shape[2] != 256: continue
            for name,groups in [('old',16),('groups4',4)]:
                add(shape,name,'na_int8_attention_bench',*shape,w,t,groups)
            if args.suite == 'attention':
                jobs.append((shape,'mlx',[str(args.mlx),'attention',*map(str,shape),str(w),str(t),'0']))
        for d in ((64,128,256) if args.suite == 'attention' else []):
            shape=(1024,1024,d,1,8,8)
            add(shape,'causal','na_int8_attention_bench',*shape,w,t,16,'causal')
    else:
        shapes = [(s,s,d,1,8,8) for d in (64,128,256) for s in (256,1024,4096)]
        shapes += [(r,c,d,1,8,8) for d in (64,128) for r,c in ((128,4096),(4096,128),(1001,2053))]
        shapes += [(1024,1024,d,b,hq,hk) for d in (64,128,256)
                   for b,hq,hk in ((4,8,8),(1,24,4),(4,16,4))]
        shapes += [(8192,8192,128,1,8,8)]
        if args.suite == 'tune-backward':
            for shape in shapes:
                if shape[0] == 8192 or shape[2] == 256: continue
                for groups,bypass in ((4,False),(8,False),(4,True),(6,True)):
                    add(shape,f'g{groups}-b{int(bypass)}','na_attention_backward_bench',*shape,w,t,
                        0,0,0,0,0,0,groups,groups,int(bypass),int(bypass))
        if args.suite == 'backward-supported':
            shapes = [shape for shape in shapes if shape[2] <= 128]
        for shape in (shapes if args.suite in ('backward','backward-int8','backward-supported') else []):
            if args.suite in ('backward','backward-supported'):
                add(shape,'na','na_attention_backward_bench',*shape,w,t)
            add(shape,'int8','na_int8_attention_backward_probe',*shape,w,t)
            if args.suite in ('backward','backward-supported'):
                jobs.append((shape,'mlx',[str(args.mlx),'backward',*map(str,shape),str(w),str(t),'0']))
    if args.shape:
        jobs = [job for job in jobs if 'x'.join(map(str,job[0])) in args.shape]
        if not jobs: p.error('no jobs match --shape')
    with (args.output/'results.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['repeat','shape','variant','metric','median_ms','exit_code'])
        failed=False
        for repeat in range(args.repeats):
            for i,(shape,variant,command) in enumerate(jobs[::1 if repeat%2==0 else -1]):
                name=f"{repeat}-{'x'.join(map(str,shape))}-{variant}"
                print(f'{i+1}/{len(jobs)} {name}',flush=True)
                env=dict(os.environ)
                env['CCV_NA_WARMUP_SECONDS']=str(args.warmup_seconds)
                if args.suite in ('backward','tune-backward','backward-int8','backward-supported'):
                    env['CCV_NA_DUMP']=str(args.output/(name+'.gradient'))
                if args.suite in ('int8','int8-long'): env['CCV_NA_INT8_ONLY']='1'
                try:
                    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=180)
                    stdout,stderr,code=result.stdout,result.stderr,result.returncode
                except subprocess.TimeoutExpired as e:
                    stdout,stderr,code=(e.stdout or b'').decode(),(e.stderr or b'').decode(),124
                (args.output/(name+'.txt')).write_text('command: '+' '.join(command)+'\n'+stdout+stderr)
                for line in stdout.splitlines():
                    match=re.search(r'^(.*?)\s+avg_ms=.*?median_ms=([\d.]+)',line)
                    if not match: match=re.search(r'^(.*?)\s+gpu_median_ms=([\d.e+-]+)',line)
                    if match: writer.writerow([repeat,'x'.join(map(str,shape)),variant,*match.groups(),code])
                if args.suite in ('backward','backward-supported') and code==0:
                    stem=f"{repeat}-{'x'.join(map(str,shape))}"
                    for impl in ('na','int8'):
                        errors=[]
                        for g in range(3):
                            ref=args.output/(stem+'-mlx.gradient.'+str(g))
                            test=args.output/(stem+'-'+impl+'.gradient.'+str(g))
                            if not ref.exists() or not test.exists(): break
                            a,b=ref.read_bytes(),test.read_bytes()
                            if len(a)!=len(b): raise RuntimeError('gradient size mismatch')
                            count=len(a)//2
                            ids=sorted(set([0,count-1]+[i*(count-1)//16383 for i in range(16384)]))
                            av=[struct.unpack_from('e',a,2*j)[0] for j in ids]
                            bv=[struct.unpack_from('e',b,2*j)[0] for j in ids]
                            diff=[x-y for x,y in zip(av,bv)]
                            l2=math.sqrt(sum(x*x for x in diff)/max(1e-30,sum(x*x for x in av)))
                            errors.append(f'gradient={g} samples={len(ids)} max_abs={max(map(abs,diff)):.8g} normalized_l2={l2:.8g}')
                            if not math.isfinite(l2) or l2 > 0.05:
                                failed=True
                                writer.writerow([repeat,'x'.join(map(str,shape)),impl,'gradient-validation-failed','',4])
                        if len(errors)==3:
                            (args.output/(stem+'-'+impl+'-accuracy.txt')).write_text('\n'.join(errors)+'\n')
                if code:
                    failed=True
                    writer.writerow([repeat,'x'.join(map(str,shape)),variant,'failed','',code])
                    print(f'FAILED {code}: {name}',flush=True)
                f.flush()
        return int(failed)


if __name__=='__main__':
    raise SystemExit(main())

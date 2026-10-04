#!/usr/bin/env python3
"""Serial CCV/actual-MLX sweeps with correctness, provenance and regression checks."""
import argparse
import csv
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time
from na_benchmark_manifest import manifest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command_text(args):
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else 'unavailable'


def metric(stdout, key):
    m = re.search(r'(?<![\w])' + re.escape(key) + r'=([^\s]+)', stdout)
    return m.group(1) if m else None


def gradient_error(reference, candidate):
    if not reference.exists() or not candidate.exists():
        raise ValueError('missing MLX or CCV VJP output')
    size = reference.stat().st_size
    if not size or size % 2 or candidate.stat().st_size != size:
        raise ValueError('VJP output size mismatch')
    count = size // 2
    # Both probes scan every gradient for finiteness. Sample numerical agreement
    # at 16k evenly spaced positions, including both ends, without loading GiBs.
    ids = sorted({i * (count - 1) // 16383 for i in range(16384)})
    error = norm = maximum = 0.0
    with reference.open('rb') as a, candidate.open('rb') as b:
        with mmap.mmap(a.fileno(), 0, access=mmap.ACCESS_READ) as av, mmap.mmap(b.fileno(), 0, access=mmap.ACCESS_READ) as bv:
            for i in ids:
                x, y = struct.unpack_from('e', av, 2*i)[0], struct.unpack_from('e', bv, 2*i)[0]
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError('nonfinite VJP output')
                delta = x-y
                error += delta*delta
                norm += x*x
                maximum = max(maximum, abs(delta))
    return dict(samples=len(ids), max_abs=maximum, normalized_l2=math.sqrt(error/max(norm, 1e-30)))


def aggregate(rows):
    groups = {}
    for row in rows:
        key = (row['id'], row['variant'])
        groups.setdefault(key, []).append(row)
    result = {}
    for key, group in groups.items():
        ok = all(r['ok'] for r in group)
        result[key] = dict(ok=ok, count=len(group),
                           median_ms=statistics.median(r['median_ms'] for r in group) if ok else None,
                           samples=[r['median_ms'] for r in group] if ok else [])
    return result


def expected_keys(meta):
    keys = set()
    for repeat in range(meta['repeats']):
        for case in meta['workloads']:
            for variant in case['variants']:
                if meta['variant_filter'] and variant not in meta['variant_filter']:
                    continue
                if variant == 'mlx' and 'mlx' not in meta['binaries']:
                    continue
                keys.add((repeat, case['id'], variant))
                if variant != 'mlx' and 'previous' in meta['binaries']:
                    keys.add((repeat, case['id'], 'previous_' + variant))
    return keys


def check_complete(meta, rows):
    keys = [(r['repeat'], r['id'], r['variant']) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected_keys(meta):
        raise ValueError('incomplete or duplicate records: expected every workload, variant and repeat')


def read_completed(path):
    meta = json.loads((path/'metadata.json').read_text())
    summary = json.loads((path/'summary.json').read_text())
    rows = list(map(json.loads, (path/'records.jsonl').read_text().splitlines()))
    check_complete(meta, rows)
    if summary['records'] != len(rows) or summary['failed_records'] != sum(not r['ok'] for r in rows):
        raise ValueError('run summary does not match completed records')
    backward = {c['id'] for c in meta['workloads'] if c['operation'] == 'backward'}
    if any(r['id'] in backward and r['variant'] != 'mlx' and r['ok'] and len(r.get('vjp_errors', [])) != 3 for r in rows):
        raise ValueError('backward run has not completed its MLX VJP checks')
    return meta, rows


def compare_rows(current, baseline, relative, absolute):
    a, b = aggregate(current), aggregate(baseline)
    result = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        row = dict(id=key[0], variant=key[1], status='missing')
        if x and y:
            if not x['ok'] or not y['ok']:
                row['status'] = 'invalid'
            else:
                delta = x['median_ms'] - y['median_ms']
                row.update(current_ms=x['median_ms'], baseline_ms=y['median_ms'], ratio=x['median_ms']/y['median_ms'],
                           status='regression' if delta > max(absolute, relative*y['median_ms']) else 'pass')
        result.append(row)
    return result


def write_summary(output, rows, relative, absolute):
    agg = aggregate(rows)
    comparison = []
    for (case, variant), value in sorted(agg.items()):
        row = dict(id=case, variant=variant, **value)
        mlx = agg.get((case, 'mlx'))
        row['mlx_speedup'] = mlx['median_ms']/value['median_ms'] if mlx and mlx['ok'] and value['ok'] else None
        comparison.append(row)
    current = [r for r in rows if not r['variant'].startswith('previous_') and r['variant'] != 'mlx']
    previous = [{**r, 'variant': r['variant'][9:]} for r in rows if r['variant'].startswith('previous_')]
    regression = compare_rows(current, previous, relative, absolute) if previous else []
    summary = dict(records=len(rows), failed_records=sum(not r['ok'] for r in rows),
                   comparisons=comparison, live_baseline_comparison=regression)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fields = ['repeat','id','variant','median_ms','ok','exit_code','cpu_l2','cpu_max_abs','output_sha256','cast_fused','attention_fp16_fallback','seconds','error']
    with (output/'results.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    return summary


def selected_cases(args):
    cases = manifest()['workloads']
    if args.manifest:
        cases = json.loads(args.manifest.read_text())['workloads']
    suites = {'coverage': {'inference','coverage'}, 'all': {'inference','coverage','backward','stress'}}
    tags = suites.get(args.suite, {args.suite})
    cases = [c for c in cases if tags.intersection(c['tags']) and (not args.id or c['id'] in args.id)]
    if args.operation:
        cases = [c for c in cases if c['operation'] == args.operation]
    if not cases:
        raise ValueError('no matching cases')
    ids = [c['id'] for c in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('workload IDs must be unique and filename-safe')
    return cases


def run(args):
    cases = selected_cases(args)
    if args.list:
        for c in cases:
            print(c['id'], c['operation'], 'x'.join(map(str,c['shape'])), ','.join(c['variants']))
        print(f'{len(cases)} cases')
        return 0
    if args.output is None:
        raise ValueError('--output is required')
    if (args.no_mlx or (args.variant and 'mlx' not in args.variant)) and any(c['operation']=='backward' for c in cases):
        raise ValueError('backward requires actual MLX for VJP validation')
    if args.repeats < 1 or args.warmup < 0 or args.samples < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        raise ValueError('invalid timing parameters')
    binaries = {'current': args.frontend.resolve()}
    if not args.no_mlx:
        binaries['mlx'] = args.mlx.resolve()
    if args.baseline_frontend:
        binaries['previous'] = args.baseline_frontend.resolve()
    for p in binaries.values():
        if not p.is_file() or not os.access(p, os.X_OK):
            raise ValueError(f'not an executable: {p}')
    # Capture only GPU model/cores, not serial numbers or user account details.
    display = json.loads(command_text(['system_profiler','SPDisplaysDataType','-json']))
    devices = [{k:d.get(k) for k in ['sppci_model','sppci_cores','spdisplays_metal']} for d in display.get('SPDisplaysDataType',[])]
    meta = dict(schema=1, devices=devices, os=command_text(['sw_vers','-productVersion']),
                memory_bytes=command_text(['sysctl','-n','hw.memsize']),
                ccv_revision=command_text(['git','-C',str(ROOT),'rev-parse','HEAD']),
                ccv_diff_sha256=hashlib.sha256(command_text(['git','-C',str(ROOT),'diff','HEAD']).encode()).hexdigest(),
                mlx_revision=command_text(['git','-C',str(args.mlx_root),'rev-parse','HEAD']) if not args.no_mlx else None,
                binaries={k:dict(path=str(v),sha256=sha256(v)) for k,v in binaries.items()},
                probe_source_sha256=sha256(HERE/'na_application_bench.cpp'),
                mlx_probe_source_sha256=sha256(HERE/'mlx_na_bench.cpp'),
                runner_sha256=sha256(Path(__file__)), manifest_generator_sha256=sha256(HERE/'na_benchmark_manifest.py'),
                workloads=cases, repeats=args.repeats, warmup=args.warmup, samples=args.samples,
                warmup_seconds=args.warmup_seconds, variant_filter=args.variant)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    meta_path = output/'metadata.json'
    if meta_path.exists():
        if not args.resume:
            raise ValueError('output already contains a run; use a new directory or --resume')
        if json.loads(meta_path.read_text()) != meta:
            raise ValueError('resume requires identical binaries, source, workloads, machine and protocol')
    else:
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    records = output/'records.jsonl'
    rows = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
    checked = output/'records.checked.jsonl'
    if checked.exists():
        status = {(r['repeat'],r['id'],r['variant']):r for r in map(json.loads,checked.read_text().splitlines()) if 'vjp_errors' in r}
        rows = [status.get((r['repeat'],r['id'],r['variant']),r) for r in rows]
    done = {(r['repeat'],r['id'],r['variant']) for r in rows}
    with records.open('a') as file:
        for repeat in range(args.repeats):
            for case in (cases if repeat%2 == 0 else cases[::-1]):
                variants = []
                for v in case['variants']:
                    if args.variant and v not in args.variant or v=='mlx' and args.no_mlx:
                        continue
                    variants.append(v)
                    if v!='mlx' and args.baseline_frontend:
                        variants.append('previous_'+v)
                if repeat%2:
                    variants.reverse()
                for variant in variants:
                    if (repeat,case['id'],variant) in done:
                        continue
                    base_variant = variant[9:] if variant.startswith('previous_') else variant
                    binary = binaries['mlx' if variant=='mlx' else 'previous' if variant.startswith('previous_') else 'current']
                    args_list = [str(binary),case['operation'],*map(str,case['shape']),str(args.warmup),str(args.samples),str(int(case.get('causal',False)))]
                    if variant!='mlx':
                        args_list += [str(int(base_variant=='int8')),str(int(base_variant=='fp32')),str(case.get('dispatch_flags',0))]
                    stem = f"{repeat}-{case['id']}-{variant}"
                    env_updates = dict(CCV_NA_WARMUP_SECONDS=str(args.warmup_seconds),
                                       CCV_NA_OUTPUT_CAST=str(int(case.get('output_cast',False))),
                                       CCV_GEMM_FUSED_CAST=str(int(case.get('output_cast',False) and base_variant=='int8')))
                    if case['operation']=='backward':
                        env_updates['CCV_NA_DUMP'] = str(output/(stem+'.gradient'))
                    # Do not inherit diagnostic overrides or dump paths from the shell.
                    env = {k:v for k,v in os.environ.items() if not k.startswith(('CCV_NA_', 'CCV_GEMM_', 'MLX_SDPA_'))}
                    start = time.time()
                    try:
                        p = subprocess.run(args_list,env={**env,**env_updates},capture_output=True,text=True,timeout=args.timeout)
                        stdout,stderr,code = p.stdout,p.stderr,p.returncode
                    except subprocess.TimeoutExpired as e:
                        stdout = e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else e.stdout or ''
                        stderr = e.stderr.decode(errors='replace') if isinstance(e.stderr,bytes) else e.stderr or ''
                        code = 124
                    (output/(stem+'.log')).write_text(stdout+stderr)
                    (output/(stem+'.command.json')).write_text(json.dumps(dict(args=args_list,env=env_updates,start=start,exit_code=code),indent=2)+'\n')
                    value = metric(stdout,'gpu_median_ms')
                    ms = float(value) if value is not None else None
                    ok = code==0 and ms is not None and math.isfinite(ms) and ms>0
                    row = dict(repeat=repeat,id=case['id'],variant=variant,median_ms=ms,ok=ok,exit_code=code,
                               cpu_l2=metric(stdout,'normalized_l2'),cpu_max_abs=metric(stdout,'cpu_sample_max_abs'),
                               output_sha256=metric(stdout,'output_sha256'),cast_fused=metric(stdout,'cast_fused'),
                               attention_fp16_fallback=metric(stdout,'attention_fp16_fallback'),seconds=time.time()-start,
                               error='' if ok else 'process or numeric check failed')
                    if row['attention_fp16_fallback']=='1' and (row['cpu_l2'] is None or not math.isfinite(float(row['cpu_l2'])) or float(row['cpu_l2']) > .01):
                        row['ok']=False
                        row['error']='FP16 fallback normalized L2 exceeds 1%'
                    if case['operation']=='backward' and variant=='mlx' and (
                            metric(stdout,'primitive') != 'ScaledDotProductAttentionVJP' or
                            metric(stdout,'outputs') != '3' or metric(stdout,'stages') != '1'):
                        row['ok']=False
                        row['error']='actual MLX SDPA VJP primitive was not timed'
                    # Defer VJP pairs until all variants for this case have run.
                    rows.append(row)
                    file.write(json.dumps(row)+'\n'); file.flush()
                    print(f"{len(rows)} {stem} {ms} ms {'PASS' if row['ok'] else 'FAIL'}",flush=True)
                if case['operation']=='backward':
                    group = [r for r in rows if r['repeat']==repeat and r['id']==case['id']]
                    for row in group:
                        if row['variant']=='mlx' or 'vjp_errors' in row:
                            continue
                        errors=[]
                        try:
                            for gradient in range(3):
                                ref=output/f"{repeat}-{case['id']}-mlx.gradient.{gradient}"
                                candidate=output/f"{repeat}-{case['id']}-{row['variant']}.gradient.{gradient}"
                                errors.append(gradient_error(ref,candidate))
                            if any(e['normalized_l2']>0.05 for e in errors):
                                raise ValueError('VJP normalized L2 exceeds 5%')
                        except ValueError as e:
                            row['ok']=False; row['error']=str(e)
                        row['vjp_errors']=errors
                    # Preserve checked records atomically before removing large dumps.
                    checked=output/'records.checked.jsonl'
                    pending=checked.with_suffix('.pending')
                    pending.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    pending.replace(checked)
                    if not args.keep_gradients:
                        for row in group:
                            for gradient in range(3):
                                (output/f"{repeat}-{case['id']}-{row['variant']}.gradient.{gradient}").unlink(missing_ok=True)
    # Merge persisted VJP status when resuming an interrupted run.
    checked=output/'records.checked.jsonl'
    if checked.exists():
        status={(r['repeat'],r['id'],r['variant']):r for r in map(json.loads,checked.read_text().splitlines()) if 'vjp_errors' in r}
        rows=[status.get((r['repeat'],r['id'],r['variant']),r) for r in rows]
    pending=records.with_suffix('.pending')
    pending.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    pending.replace(records)
    check_complete(meta, rows)
    summary=write_summary(output,rows,args.relative,args.absolute_ms)
    if not rows:
        raise ValueError('no matching variants')
    bad=sum(r['status']!='pass' for r in summary['live_baseline_comparison'])
    print(f"COMPLETE: {len(rows)} records; {summary['failed_records']} failed; {bad} baseline comparisons need review")
    return int(bool(summary['failed_records'] or bad))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--suite',choices=['smoke','inference','coverage','backward','stress','all'],default='inference')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--id',action='append')
    parser.add_argument('--operation',choices=['attention','matmul','backward'])
    parser.add_argument('--variant',action='append',choices=['fp16','fp32','int8','mlx'])
    parser.add_argument('--frontend',type=Path,default=HERE/'na_application_bench')
    parser.add_argument('--baseline-frontend',type=Path)
    parser.add_argument('--mlx',type=Path,default=HERE/'mlx_na_bench')
    parser.add_argument('--mlx-root',type=Path,default=Path.home()/'workspace/mlx')
    parser.add_argument('--no-mlx',action='store_true')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--samples',type=int,default=7)
    parser.add_argument('--warmup-seconds',type=float,default=.5)
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--keep-gradients',action='store_true')
    parser.add_argument('--relative',type=float,default=.05,help='Regression relative threshold (default 5 percent)')
    parser.add_argument('--absolute-ms',type=float,default=.02,help='Regression must also exceed this absolute difference')
    parser.add_argument('--compare',nargs=2,type=Path,metavar=('BASELINE','CURRENT'),help='Compare completed runs on the same device/protocol')
    args=parser.parse_args()
    try:
        if any(not math.isfinite(v) or v < 0 for v in [args.relative, args.absolute_ms]) or not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('thresholds must be finite and nonnegative; timeout must be positive')
        if args.compare:
            baseline,current=args.compare
            (a, baseline_rows), (b, current_rows) = [read_completed(p) for p in [baseline,current]]
            for key in ['devices','os','memory_bytes','workloads','repeats','warmup','samples','warmup_seconds','variant_filter']:
                if a[key]!=b[key]:
                    raise ValueError(f'incompatible comparison: {key}; run local CCV versus MLX on each device')
            result=compare_rows(current_rows,baseline_rows,args.relative,args.absolute_ms)
            print(json.dumps(result,indent=2))
            return int(any(r['status']!='pass' for r in result))
        return run(args)
    except (ValueError,OSError) as e:
        parser.error(str(e))


if __name__=='__main__':
    sys.exit(main())

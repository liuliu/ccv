#!/usr/bin/env python3
"""Serial decode benchmarks; preserve every sample, failure and machine detail."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_text(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else 'unavailable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', type=Path, default=HERE / 'attention_r1_bench')
    parser.add_argument('--baseline', type=Path, help='Saved attention_r1_bench binary')
    parser.add_argument('--mlx', type=Path, help='MLX probe with the archived mlx_na_bench.cpp CLI (FP16 only)')
    parser.add_argument('--mlx-root', type=Path, default=Path.home() / 'workspace/mlx')
    parser.add_argument('--thermal', type=Path, default=HERE / 'attention_r1_thermal')
    parser.add_argument('--manifest', type=Path, default=HERE / 'attention_r1_shapes.json')
    parser.add_argument('--precision', choices=['fp16', 'bf16', 'both'], default='fp16')
    parser.add_argument('--id', action='append')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--warmup-seconds', type=float, default=2.0)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text())['workloads']
    if args.id:
        known = {case['id'] for case in cases}
        if set(args.id) - known:
            parser.error('unknown workload ID')
        cases = [case for case in cases if case['id'] in args.id]
    if not cases or len({case['id'] for case in cases}) != len(cases) or any(
            not re.fullmatch(r'[a-zA-Z0-9_-]+', case['id']) for case in cases):
        parser.error('empty manifest, duplicate IDs or unsafe IDs')
    if args.list:
        for case in cases:
            print(case['id'], case['shape'], 'causal=' + str(case['causal']))
        return 0
    if args.output is None or args.output.exists():
        parser.error('--output must name a new directory')
    if min(args.repeats, args.samples) < 1 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error('invalid timing parameters')
    if args.mlx and args.precision != 'fp16':
        parser.error('the MLX comparison probe accepts FP16 only; run BF16 separately')
    args.thermal = args.thermal.resolve()
    binaries = {'current': args.probe.resolve()}
    if args.baseline:
        binaries['baseline'] = args.baseline.resolve()
    if args.mlx:
        binaries['mlx'] = args.mlx.resolve()
    for binary in [*binaries.values(), args.thermal]:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f'not an executable: {binary}')
    display = json.loads(command_text(['system_profiler', 'SPDisplaysDataType', '-json']))
    metadata = dict(
        devices=[{k: d.get(k) for k in ['sppci_model', 'sppci_cores', 'spdisplays_metal']}
                 for d in display.get('SPDisplaysDataType', [])],
        os=command_text(['sw_vers', '-productVersion']),
        memory_bytes=command_text(['sysctl', '-n', 'hw.memsize']),
        mlx_revision=command_text(['git', '-C', str(args.mlx_root), 'rev-parse', 'HEAD']) if args.mlx else None,
        revision=command_text(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
        diff=command_text(['git', '-C', str(ROOT), 'diff', 'HEAD']),
        binaries={name: dict(path=str(path), sha256=digest(path)) for name, path in binaries.items()},
        runner_sha256=digest(__file__), probe_source_sha256=digest(HERE / 'attention_r1_bench.cpp'),
        manifest_sha256=digest(args.manifest), workloads=cases,
        precision=args.precision, repeats=args.repeats, samples=args.samples,
        warmup_seconds=args.warmup_seconds, cooldown_seconds=1,
        thermal_probe_sha256=digest(args.thermal))
    args.output.mkdir(parents=True)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith('MLX_SDPA_')}
    env['CCV_NA_WARMUP_SECONDS'] = str(args.warmup_seconds)
    env['CCV_NA_OUTPUT_CAST'] = '0'
    precisions = [0, 1] if args.precision == 'both' else [int(args.precision == 'bf16')]
    records = []
    for repeat in range(args.repeats):
        for case in cases if repeat % 2 == 0 else cases[::-1]:
            for precision in precisions:
                variants = [v for v in binaries if v != 'mlx' or 'mlx' in case.get('variants', ['mlx'])]
                if repeat % 2:
                    variants.reverse()
                for variant in variants:
                    deadline = time.monotonic() + 300
                    while command_text([str(args.thermal)]) != '0':
                        if time.monotonic() > deadline:
                            raise RuntimeError('GPU did not return to nominal thermal state; run is incomplete')
                        time.sleep(5)
                    common = [*map(str, case['shape']), '3', str(args.samples), str(int(case['causal']))]
                    command = [str(binaries[variant])]
                    command += ['attention', *common] if variant == 'mlx' else [
                        *common, str(case.get('dispatch_flags', 3)), str(precision)]
                    stem = f"{repeat}-{case['id']}-p{precision}-{variant}"
                    start = time.monotonic()
                    try:
                        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=600)
                        output, exit_code = result.stdout + result.stderr, result.returncode
                    except subprocess.TimeoutExpired as error:
                        # subprocess.run terminates the child on this explicit per-case limit.
                        output = (error.stdout or b'').decode() + (error.stderr or b'').decode()
                        exit_code = 'timeout'
                    (args.output / (stem + '.log')).write_text(output)
                    pattern = r'gpu_samples_ms=([^\s]+)' if variant == 'mlx' else r'stage=production samples_ms=([^\s]+)'
                    match = re.search(pattern, output)
                    samples = [float(x) for x in match[1].split(',')] if match else []
                    thermal_after = command_text([str(args.thermal)])
                    ok = (exit_code == 0 and len(samples) == args.samples and
                          all(math.isfinite(x) and x > 0 for x in samples) and
                          'all_finite=1' in output and thermal_after == '0')
                    row = dict(repeat=repeat, id=case['id'], precision=precision, variant=variant,
                               command=command, exit_code=exit_code, samples_ms=samples, ok=ok,
                               median_ms=statistics.median(samples) if samples else None,
                               seconds=time.monotonic() - start, thermal_before='0', thermal_after=thermal_after)
                    records.append(row)
                    with (args.output / 'records.jsonl').open('a') as file:
                        file.write(json.dumps(row) + '\n')
                    print(stem, row['median_ms'], 'PASS' if ok else 'FAIL', flush=True)
                    time.sleep(1)
    comparisons = []
    for case in cases:
        for precision in precisions:
            times = {}
            for variant in binaries:
                group = [r for r in records if r['id'] == case['id'] and
                         r['precision'] == precision and r['variant'] == variant]
                if len(group) == args.repeats and all(r['ok'] for r in group):
                    times[variant] = statistics.median(r['median_ms'] for r in group)
            row = dict(id=case['id'], precision=precision, medians_ms=times)
            if 'current' in times:
                for reference in ['baseline', 'mlx']:
                    if reference in times:
                        row[reference + '_ratio'] = times['current'] / times[reference]
            row['regression'] = ('baseline' in times and 'current' in times and
                                 times['current'] > times['baseline'] * 1.05 and
                                 times['current'] - times['baseline'] > 0.002)
            comparisons.append(row)
    failed = sum(not r['ok'] for r in records)
    summary = dict(records=len(records), failed=failed, comparisons=comparisons)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return int(bool(failed or any(row['regression'] for row in comparisons)))


if __name__ == '__main__':
    raise SystemExit(main())

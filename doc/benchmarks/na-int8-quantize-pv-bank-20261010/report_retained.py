"""Retained candidate report with per-workload diagnostic comparisons."""
import argparse,hashlib,json,re,statistics
from pathlib import Path
from na_benchmark_manifest import manifest
here=Path(__file__).resolve().parent
repo=here.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,required=True,help='Directory containing local raw benchmark stages')
parser.add_argument('--output',type=Path,default=here/'retained-results.txt')
parser.add_argument('--json',type=Path,help='Optional machine-readable report outside the repository')
args=parser.parse_args()
raw=args.input.resolve()
cases=[c for c in manifest()['workloads'] if c['operation']=='attention']
report=dict(scope='All 128 eligible forward attention workloads in the recovered bank, FP16 I/O',
    default='quantizePV=true is the default; quantizePV=false is forward INT8 QK with floating PV. Shared correctness fixes are documented in validation-summary.txt.',
    diagnostic_reference=.75,
    threshold_note='75% throughput retention is a diagnostic reference, not a hard pass/fail gate. Compare NAInt8Attention false against true; NAAttention speedup is reported separately.',
    mac_note='Mac results use isolated GPU measurements; contended measurements are excluded.',
    source_hashes={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'lib/nnc/mfa/kernels').glob('NAInt8Attention*.*') if p.suffix in ('.cpp','.hpp')},devices={})
lines=[report['scope'],report['default'],'Per workload: speed(false) / speed(true) = true_ms / false_ms; preparation included.',report['threshold_note'],report['mac_note'],'Ordinary bank: 15 rotating samples; Mac decode: 31 samples; H3: 3 samples. Raw timings and thermal states are retained locally, outside the repository. Source paths below are relative to that raw-results directory.',
       'Nine rows use measurements with the R/C crossover selector; six Mac decode rows have separate repeated measurements. The remaining rows retain verified source-equivalent measurements.','']
for device,stages in [('mac',['mac-final-long','mac-final-bank','mac-no-split-decode','mac-single-cutoff','mac-rc-cutoff','mac-decode-final']),('ipad',['ipad-final-long','ipad-final-bank','ipad-final-pairs','ipad-no-split-decode','ipad-single-cutoff','ipad-rc-cutoff'])]:
 rows=[]
 for case in cases:
  found=[raw/stage/(case['id']+'.json') for stage in stages if (raw/stage/(case['id']+'.json')).exists()]
  if not found:
   rows.append(dict(case=case,status='unmeasured'));continue
  path=found[-1];x=json.loads(path.read_text());timing=next((t for t in x.get('timings',[]) if t['full']==1),None)
  row=dict(case=case,source=str(path.relative_to(raw)),correct=x['correct'],exit=x['exit'],thermal=x.get('thermal',[]),timing=timing)
  log=path.with_suffix('.log').read_text();m=re.search(r'timing full=1 .*samples=(\d+)',log);row['samples']=int(m[1]) if m else None
  if timing:
   row['retention']=timing['int8_ms']/timing['hybrid_ms'];row['na_speedup']=timing['na_ms']/timing['hybrid_ms'] if timing['na_ms'] is not None else None
   row['below_reference']=row['retention']<.75;row['na_faster']=row['na_speedup']>1 if row['na_speedup'] is not None else None
   samples=[tuple(map(float,m)) for m in re.findall(r'sample full=1 round=\d+ int8_ms=(\S+) hybrid_ms=(\S+) na_ms=(\S+)',log)]
   if samples:
    row['paired_retention_median']=statistics.median(s[0]/s[1] for s in samples)
    row['sample_ranges_ms']={name:[min(s[i] for s in samples),max(s[i] for s in samples)] for i,name in enumerate(['true','false','na'])}
   row['status']=('flagged_below_75_percent' if row['below_reference'] else 'measured') if row['correct'] else 'incorrect'
  else:row['status']='failed_or_unmeasured'
  rows.append(row)
 counts={status:sum(r['status']==status for r in rows) for status in sorted({r['status'] for r in rows})}
 measured=[r for r in rows if r.get('correct') and r.get('timing')]
 summary=dict(measured=len(measured),median_retention=statistics.median(r['retention'] for r in measured) if measured else None,
     minimum_retention=min((r['retention'] for r in measured),default=None),faster_than_na=sum(r.get('na_faster') is True for r in measured),
     na_comparisons=sum(r.get('na_speedup') is not None for r in measured),
     median_na_speedup=statistics.median(r['na_speedup'] for r in measured if r.get('na_speedup') is not None) if any(r.get('na_speedup') is not None for r in measured) else None,
     non_nominal_thermal_cases=sum(any(t!='0' for t in r['thermal']) for r in measured))
 report['devices'][device]=dict(counts=counts,summary=summary,completed_15_sample_cases=sum((r.get('samples') or 0)>=15 and r.get('correct',False) for r in rows),rows=rows)
 lines += [device.upper()+' '+json.dumps(counts),'case | true ms | false ms | NA ms | retention | NA speedup | samples | status | source']
 for row in rows:
  t=row.get('timing');key=row['case']['id']
  if t:
   na_time=f"{t['na_ms']:.6f}" if t['na_ms'] is not None else 'unavailable'
   na_speed=f"{row['na_speedup']:.3f}x" if row['na_speedup'] is not None else 'unavailable'
   lines.append(f"{key} | {t['int8_ms']:.6f} | {t['hybrid_ms']:.6f} | {na_time} | {row['retention']:.3%} | {na_speed} | {row['samples']} | {row['status']} | {row['source']}")
  else:lines.append(f"{key} | unavailable | {row['status']} | {row.get('source','')}")
 lines.append('')
report['all_workloads_measured']=all(r.get('correct') and r.get('timing') for d in report['devices'].values() for r in d['rows'])
report['decode_repeats']=[]
lines += ['MAC DECODE INDEPENDENT REPEAT',
          '31 rotating samples per variant, preparation included. The first run is retained in the bank above; this repeat checks timing stability.',
          'case | true ms | false ms | NA ms | retention | NA speedup | correct | source']
for case in cases:
 path=raw/'mac-decode-repeat'/(case['id']+'.json')
 if not path.exists():continue
 x=json.loads(path.read_text());t=next((t for t in x.get('timings',[]) if t['full']==1),None)
 record=dict(case=case['id'],source=str(path.relative_to(raw)),correct=x['correct'],exit=x['exit'],timing=t)
 report['decode_repeats'].append(record)
 if t:
  lines.append(f"{case['id']} | {t['int8_ms']:.6f} | {t['hybrid_ms']:.6f} | {t['na_ms']:.6f} | {t['int8_ms']/t['hybrid_ms']:.3%} | {t['na_ms']/t['hybrid_ms']:.3f}x | {x['correct']} | {record['source']}")
 else:lines.append(f"{case['id']} | unavailable | exit {x['exit']} | {record['source']}")
lines.append('')
report['crosschecks']={}
lines += ['DIRECT TRUE/FALSE CROSSCHECKS',
          'These omit NAAttention to check order-sensitive outliers. They supplement, rather than replace, the bank results above.']
for device,stage in [('mac','mac-pair-crosscheck'),('ipad','ipad-pair-crosscheck')]:
 records=[]
 for path in sorted((raw/stage).glob('*.json')):
  if path.name=='metadata.json':continue
  x=json.loads(path.read_text())
  timing=next((t for t in x.get('timings',[]) if t['full']==1),None)
  log=path.with_suffix('.log').read_text()
  m=re.search(r'timing full=1 .*samples=(\d+)',log)
  record=dict(case=x['case']['id'],source=str(path.relative_to(raw)),correct=x['correct'],exit=x['exit'],thermal=x.get('thermal',[]),timing=timing,samples=int(m[1]) if m else None)
  if timing:
   record['retention']=timing['int8_ms']/timing['hybrid_ms']
   pairs=[tuple(map(float,p)) for p in re.findall(r'pair_sample full=1 round=\d+ int8_ms=(\S+) hybrid_ms=(\S+)',log)]
   if pairs:record['paired_retention_median']=statistics.median(a/b for a,b in pairs)
   lines.append(f"{device} {record['case']} | true {timing['int8_ms']:.6f} ms | false {timing['hybrid_ms']:.6f} ms | retention {record['retention']:.3%} | samples {record['samples']}")
  else:lines.append(f"{device} {record['case']} | unavailable | exit {record['exit']}")
  records.append(record)
 report['crosschecks'][device]=records
lines.append('')
report['validation']=dict(default_source='See validation-summary.txt for exact-source comparison and the shared quantizer bounds fix',
    bank_source='2856 forward configurations from 119 workloads match their measured source; nine workloads have measurements with the R/C selector, and six Mac decode workloads have independent repeated measurements',
    build='make debug -j4 passed',
    integration={'mpsblas_attention':'40/40 passed'},
    cuda='Intermediate-precision rule source-checked; compiler and hardware unavailable locally',
    quantizer_boundaries='ipad-quantizer-boundaries-fixed: 9 Metal validation checks plus 9 CPU-reference checks passed')
report['features']={}
for device,stage in [('mac','mac-features-fixed'),('ipad','ipad-features-final')]:
 records=[dict(source=str(p.relative_to(raw)),**json.loads(p.read_text())) for p in sorted((raw/stage).glob('*.json'))]
 if not records:continue
 report['features'][device]=dict(total=len(records),floating_pv_pass=sum(r.get('false_correct',False) for r in records),records=records)
 lines += [device.upper()+' FEATURE CHECKS: '+str(sum(r.get('false_correct',False) for r in records))+'/'+str(len(records))+' floating-PV checks passed.',
           'These historical feature runs include default-mode errors on strong sinks and packed batched masks. The packed-mask offset fix now applies to both modes; see validation-summary.txt for subsequent validation.']
 for record in records:
  if not record.get('false_correct'):lines.append('  Unavailable or failed: '+record['source']+' exit='+str(record['exit']))
lines += ['', 'Additional D=8 diagnostic: both true and false preparation pass, but both attention commands stall. This additional head-size diagnostic is outside the original bank.']
lines.insert(0,'COVERAGE: '+', '.join(device+' '+str(value['summary']['measured'])+'/128 measured' for device,value in report['devices'].items())+'; unavailable cases are reported explicitly.')
if args.json:args.json.write_text(json.dumps(report,indent=2))
args.output.write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:7]))
for device,value in report['devices'].items():print(device,value['counts'])

"""Compare generated source and dispatch against tip or a saved source snapshot.

Copies both source generators, namespaces their symbols, and omits only their
Metal-library compilation. All descriptor selection and source generation run
unchanged. No production files are rewritten.
The shared backward quantizer bounds fix is checked as an explicit substitution;
all other generated source must match byte for byte.
"""
import concurrent.futures
import argparse
import re
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-commit', default='4b8d56920b9a863c0e4ecb2058cb77cc587515b7', help='Tip before this change; keep the comparison reproducible after commits')
parser.add_argument("--reference", type=str, help="Saved source directory for false-path noncausal equivalence")
parser.add_argument("--all-forward", action="store_true")
parser.add_argument("--core-count", type=int)
parser.add_argument('--filter', default='', help='Regular expression selecting workload IDs')
parser.add_argument("--archive", default='/private/tmp/ccv-quantize-pv-mac/libmfa.a', help='libmfa.a produced by build.py')
args=parser.parse_args()
from pathlib import Path
import subprocess
from na_benchmark_manifest import manifest
here=Path(__file__).resolve().parent
repo=here.parents[2]
out=Path('/private/tmp/ccv-quantize-pv-source-check' + ('-false' if args.reference else ''))
out.mkdir(exist_ok=True)
kernels=repo/'lib/nnc/mfa/kernels'
sources=[]
for prefix in ['Tip','Snapshot']:
 for stem in ['NAInt8AttentionDescriptor','NAInt8AttentionKernelDescriptor','NAInt8AttentionKernel']:
  for suffix in ['.hpp','.cpp']:
   source=kernels/(stem+suffix)
   text=((Path(args.reference)/source.name).read_text() if args.reference else subprocess.check_output(['git','show',args.base_commit+':'+str(source.relative_to(repo))],cwd=repo,text=True)) if prefix=='Tip' else source.read_text()
   text=text.replace('NAInt8Attention',prefix+'NAInt8Attention').replace('NAINT8ATTENTION',prefix.upper()+'NAINT8ATTENTION')
   text=text.replace('"../ccv_nnc_mfa.hpp"','"'+str(repo/'lib/nnc/mfa/ccv_nnc_mfa.hpp')+'"')
   if stem=='NAInt8AttentionKernel' and suffix=='.cpp':
    start=text.index('  auto string = NS::String::string(source.c_str()')
    end=text.index('  CCV_NNC_MFA_CHECK_ERROR(error);',start)+len('  CCV_NNC_MFA_CHECK_ERROR(error);')
    text=text[:start]+text[end:]
   target=out/(prefix+stem+suffix)
   target.write_text(text)
   if suffix=='.cpp': sources.append(target)
flags=['xcrun','clang++','-std=c++17','-O2','-fblocks','-w','-DHAVE_MPS','-I'+str(repo/'lib'),'-I'+str(kernels),'-I'+str(out)]
def compile(source):
 obj=source.with_suffix('.o')
 subprocess.run(flags+['-c',str(source),'-o',str(obj)],check=True)
 return str(obj)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor: objects=list(executor.map(compile,sources))
verifier=(here/'verify_default.cpp').read_text()
if args.core_count:
 verifier=verifier.replace('{10u,40u}', '{'+str(args.core_count)+'u}')
if args.reference:
 verifier=verifier.replace('mode<3', 'mode<1').replace('if (!a.quantizePV) return 2;', 'a.quantizePV=z.quantizePV=false;').replace('default_source_and_dispatch_identical=', 'false_source_and_dispatch_identical=')
(out/'verify.cpp').write_text(verifier)
subprocess.run(flags+[str(out/'verify.cpp'),*objects,args.archive,'-framework','Foundation','-framework','Metal','-framework','QuartzCore','-framework','Accelerate','-framework','MetalPerformanceShaders','-framework','IOKit','-o',str(out/'verify')],check=True)
(out/'cases.txt').write_text(''.join(' '.join(map(str,c['shape']+[int(c['causal']),c['dispatch_flags']]))+'\n' for c in manifest()['workloads'] if c['operation']=='attention' and re.search(args.filter,c['id']) and (not args.reference or args.all_forward or not c['causal'])))
subprocess.run([str(out/'verify'),str(out/'cases.txt')],check=True)

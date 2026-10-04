import json,statistics,sys
from pathlib import Path
for arg in sys.argv[1:]:
 p=Path(arg);groups={}
 print('\n'+str(p))
 for r in map(json.loads,p.read_text().splitlines()):
  groups.setdefault(r['id'],{}).setdefault(r['arm'],[]).append(r)
 for name,arms in groups.items():
  stats={}
  for a,rs in arms.items():
   xs=[float(r['median_ms']) for r in rs]
   mean=statistics.mean(xs)
   stats[a]={'mean_ms':mean,'spread_pct':100*(max(xs)-min(xs))/mean,'n':len(xs)}
  print(name+' | '+' | '.join(a+f" {s['mean_ms']:.4f} ms (spread {s['spread_pct']:.1f}%, n={s['n']})" for a,s in stats.items()))
  if 'base' in stats:
   print('  changes vs parent: '+', '.join(a+f" {100*(s['mean_ms']/stats['base']['mean_ms']-1):+.1f}%" for a,s in stats.items() if a!='base'))

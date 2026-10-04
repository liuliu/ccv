#!/usr/bin/env python3
"""Check that failed/partial runs cannot become speedups and VJP dumps are checked."""
import math
from pathlib import Path
import struct
import tempfile
import unittest
from na_application_sweep import aggregate, compare_rows, gradient_error, check_complete
from na_benchmark_manifest import manifest


def row(case='a', variant='fp16', ms=1.0, ok=True):
    return dict(id=case, variant=variant, median_ms=ms, ok=ok)


class BenchmarkChecks(unittest.TestCase):
    def test_failed_repeat_invalidates_fast_time(self):
        result=aggregate([row(ms=1),row(ms=.1,ok=False)])
        self.assertFalse(result[('a','fp16')]['ok'])
        self.assertIsNone(result[('a','fp16')]['median_ms'])
        self.assertEqual(compare_rows([row(ms=.1,ok=False)],[row()],.05,.02)[0]['status'],'invalid')

    def test_missing_cases_and_noise_floor(self):
        self.assertEqual(compare_rows([row('a')],[row('b')],.05,.02)[0]['status'],'missing')
        self.assertEqual(compare_rows([row(ms=.011)],[row(ms=.01)],.05,.02)[0]['status'],'pass')
        self.assertEqual(compare_rows([row(ms=1.2)],[row()],.05,.02)[0]['status'],'regression')

    def test_completely_missing_case_is_rejected(self):
        meta=dict(repeats=1, variant_filter=None, binaries={'current':{},'mlx':{}},
                  workloads=[dict(id='a',variants=['fp16','mlx']),dict(id='b',variants=['fp16','mlx'])])
        rows=[dict(row('a',v),repeat=0) for v in ['fp16','mlx']]
        with self.assertRaises(ValueError):check_complete(meta,rows)
        rows.extend(dict(row('b',v),repeat=0) for v in ['fp16','mlx'])
        check_complete(meta,rows)
        with self.assertRaises(ValueError):check_complete(meta,rows+rows[:1])

    def test_vjp_size_finiteness_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            a.write_bytes(struct.pack('4e',1,2,3,4));b.write_bytes(a.read_bytes())
            self.assertEqual(gradient_error(a,b)['normalized_l2'],0)
            b.write_bytes(struct.pack('4e',1,2,3,8))
            self.assertGreater(gradient_error(a,b)['normalized_l2'],.5)
            b.write_bytes(struct.pack('4e',1,2,3,math.nan))
            with self.assertRaises(ValueError):gradient_error(a,b)
            b.write_bytes(b'')
            with self.assertRaises(ValueError):gradient_error(a,b)
            b.unlink()
            with self.assertRaises(ValueError):gradient_error(a,b)

    def test_manifest_has_exact_long_workloads_and_unique_requests(self):
        cases=manifest()['workloads']
        keys=[(c['operation'],tuple(c['shape']),c['dispatch_flags'],c['causal'],c['output_cast']) for c in cases]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertGreater(len(cases),400)
        for r in [31018,70186]:
            self.assertTrue(any(c['operation']=='attention' and c['shape']==[r,r,128,1,56,56] for c in cases))
        self.assertTrue(any(c['operation']=='backward' and c['shape'][2]==256 for c in cases))


if __name__=='__main__':unittest.main()

# MoE routing launch sizes

Measured on Apple M5 Max, 2026-09-11. The 384-expert/top-6 capacity patch is
correct with either 384 or 512 threads. At the model's hidden width of 5120,
512 threads are faster, so retain the existing 256/512 policy.

The kernel uses one threadgroup. Each thread owns one expert candidate during
top-k selection, then participates in copying the activation. The latter work
also benefits from threads that have no expert candidate. A power-of-two size
is not a requirement of the reduction; the measured benefit applies to the
current fused selection-and-gather implementation.

## Method

`bin/mfa/moe_routing_bench.cpp` uses the production `MoERoutingKernel` source,
compiling separate pipelines with matching reduction scratch for each launch:

- Bucketed: 256 threads through 256 experts, otherwise 512.
- Aligned: expert count rounded up to a multiple of 32, without a minimum.

The harness checks every output buffer for bit-identical results between the
two policies before timing. It warms both pipelines for 10 pairs, alternates
their execution order each pair, and measures GPU command-buffer duration
over 256 serial dispatches. Compilation and host encoding time are outside
the GPU interval. Results include GPU dispatch overhead and represent repeated
routing dispatches, not complete-model latency.

Inputs are one token, top-6, FP32 logits/bias, and FP16 activations. Both biased
top-k selection and preselected IDs are tested, each with six copied activation
rows (`expanded`) or one (`single`). The initial sweep uses 80 timed pairs;
the focused repeat uses 160. Equal-size controls at E=256, 511, and 512 remain
close to parity. `paired_speedup` in the output is the median of each pair's
bucketed/aligned time ratio; it can differ from the ratio of separate medians.

## Focused 384-expert repeat, H=5120

GPU microseconds per dispatch:

| Routing | Activation copy | 512 threads | 384 threads | Increase with 384 |
| --- | --- | ---: | ---: | ---: |
| Top-6 selection | Six rows | 6.5690 | 7.8280 | 19.2% |
| Top-6 selection | One row | 3.7430 | 4.1318 | 10.4% |
| Preselected IDs | Six rows | 5.3254 | 6.5615 | 23.2% |
| Preselected IDs | One row | 2.3960 | 2.8619 | 19.4% |

The initial sweep independently measured 6.5177 vs 7.7882 microseconds for
top-6 with six copied rows, and 3.7504 vs 4.0752 with one copied row.

## Other expert counts, H=5120

Initial sweep, top-6 selection and six copied rows:

| Experts | Bucketed threads | Aligned threads | Bucketed µs | Aligned µs |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 256 | 32 | 10.1528 | 63.0977 |
| 128 | 256 | 128 | 10.1273 | 17.7753 |
| 256 | 256 | 256 | 10.2126 | 10.2381 |
| 257 | 512 | 288 | 6.5107 | 9.4783 |
| 384 | 512 | 384 | 6.5177 | 7.7882 |
| 385 | 512 | 416 | 6.5293 | 7.4916 |
| 512 | 512 | 512 | 6.5728 | 6.5384 |

Reducing the activation width to H=512 largely removed the top-6 difference:
paired speedups were 0.9918 for expanded and 0.9802 for single-row output.
The H=1 diagnostic also had top-6 paired speedups near 1, but absolute timings
were noisy. These probes and the preselected-ID cases support activation
copying as the main reason for the H=5120 advantage; they do not isolate every
compiler or scheduling effect. Do not compare absolute timings across these
different-width runs as a pure copy-cost subtraction.

## Reproduce

From `bin/mfa`:

```sh
make -j4 moe_routing_bench
./moe_routing_bench 80 256
./moe_routing_bench 160 256 384
./moe_routing_bench 80 256 384 512
./moe_routing_bench 160 256 384 1
```

The final arguments optionally select expert count (0 runs the sweep) and
hidden width (default 5120). GPU access outside the restricted sandbox is
required in this workspace.

## Correctness

The aligned implementation, retaining a 256-thread minimum, passed all six
`moe_routing.tests` cases. The extended boundary case covers 18 expert-count
entries and five modes, including actual top-32 selection, preselected top-32,
FP16/FP32 routing, FP16/BF16 activations, cache reuse, and the 513-expert graph
fallback. Both benchmark policies produced bit-identical outputs in all 40
sweep configurations and the focused repeats.

The final production 256/512 policy also passed all six cases after rebuilding
with `make -j4 DEBUG=1 moe_routing.tests` in `test/int/nnc` and running
`./moe_routing.tests`. No cases were skipped.

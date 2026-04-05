# Tensor-Op Apple Skill

## Purpose

Use this skill when working on Apple Metal tensor-op kernels that target the Metal Performance Primitives `tensor_ops` / `matmul2d` path on Apple GPUs and the on-GPU Neural Accelerators. This is **not** the Apple Neural Engine.

This skill is for kernels like:

- `NAAttention`
- `NAInt8Attention`
- `NAMatMul`
- `NAInt8MatMul`
- similar MFA kernels that use:
  - cooperative tensors
  - `matmul2d`
  - threadgroup staging
  - launch-order tuning
  - mixed precision
  - quantization / dequantization

The main lesson from this workspace is:

- performance work on Apple tensor-op kernels is tightly coupled to correctness
- many “fast” paths are only fast because they are wrong
- the best kernels usually come from reducing live state and fixing data layout / scale semantics, not from blindly increasing tile sizes

## When To Use This Skill

Use this skill when you need to:

- tune tile sizes for a tensor-op kernel
- decide whether to use threadgroup staging or direct device loads
- debug a performance cliff at a specific shape
- debug correctness regressions caused by a staging or quantization change
- choose launch order, simdgroup count, or staging strategy
- design a benchmark methodology for Apple tensor-op kernels

Do not use this skill for:

- MPSGraph-first operator work that does not use MFA tensor-op kernels
- Apple Neural Engine work
- CPU or CUDA tuning

## Mental Model

Apple tensor-op performance on these kernels is usually dominated by four things:

1. Live state pressure
2. Threadgroup memory footprint
3. Whether the reduction math respects quantization-scale boundaries
4. Launch / traversal structure interacting with cache locality

The core mistake to avoid is assuming the best standalone GEMM configuration will also be best for fused attention. In attention-like kernels, the softmax bridge, reductions, and correction terms inflate live state. That changes the winning point.

## GEMM vs Fused Attention

`NAMatMul` and `NAInt8MatMul` are simpler than `NAAttention` and `NAInt8Attention`, but that simplicity is exactly why they matter as tuning references.

For GEMM-like kernels:

- the launch grid is usually a real 2D tiled output problem
- locality across neighboring output tiles is easier to exploit
- staging policy is easier to reason about
- quantization is usually tied directly to `A/B` packing and the reduction axis
- there are fewer live correction terms competing for registers

For fused attention-like kernels:

- the kernel is no longer just GEMM
- softmax reconstruction, `L/D`, correction terms, and extra traversals inflate live state
- a standalone GEMM win often shrinks or disappears
- some tricks that are robust in GEMM become fragile in backward attention

General rule:

- treat standalone GEMM wins as an upper bound on what a fused kernel might recover
- do not directly port a winning `NAMatMul` or `NAInt8MatMul` shape into attention and assume it will survive
- but do reuse the matmul-side mental model for:
  - launch locality
  - staging discipline
  - reduction-axis reasoning
  - baseline fairness

## First Principles

### 1. Correctness comes before speed

A path that is not correctness-cleared is not benchmarkable.

In this workspace, several “wins” were fake because:

- a gradient collapsed toward zero
- a benchmark selector was stale
- a path mixed incompatible quantization scales inside a reduction
- a probe was comparing the wrong layout

Always validate:

- max absolute diff
- max relative diff
- reference output max magnitude
- test output max magnitude

Why all four matter:

- diff metrics catch local errors
- magnitude metrics catch collapse-to-zero or explosion

This was critical for:

- dense backward staged-path debugging
- `NAInt8Attention` backward `dV` debugging

### 2. Separate selector questions from kernel questions

When a shape cliff appears, determine whether it is caused by:

- a real kernel cliff
- a stale benchmark selector
- a production selector mismatch

Use an isolated bench that can force:

- bypass vs staged
- query vs key/value behavior
- specific simdgroup counts
- specific block sizes

Only after that should you trust the end-to-end compare bench.

### 3. Simdgroup math tiles and threadgroup packaging are different knobs

Do not collapse them mentally into one “tile size”.

The MPP guide and this workspace both support this:

- larger math tiles can be bad
- while larger reuse / packaging can still be good

For attention-style kernels, these often diverge:

- `blockR = 16` stayed robust
- larger traversal tiles like `blockC = 128` or `256` were often catastrophic
- but smarter packaging / launch order could still help

## Practical Rules That Held Up

### 1. `blockR = 16` is the default until proven otherwise

For both dense and int8 attention work:

- `blockR = 16` consistently survived
- `blockR = 8` and `blockR = 32` were generally worse

If you are starting a new sweep, start from `blockR = 16`.

### 2. Bigger traversal is usually not the fix for fused attention

In fused attention-like kernels, larger traversal tiles:

- increase score-tile size
- inflate reduction state
- increase register pressure
- often hurt much more than they help

Observed patterns:

- `blockC = 64` was usually the stable winner
- `blockC = 128` and `256` were often much worse

Treat larger traversal as a special experiment, not a default tuning direction.

### 2a. For standalone GEMM-like kernels, larger packaging can stay good longer

For `NAMatMul` and `NAInt8MatMul`, the simpler dataflow means:

- threadgroup reuse is easier to cash in
- locality-oriented launch order is more predictably helpful
- the winning point can tolerate more aggressive packaging than fused attention

That does **not** mean “always make tiles bigger.” It means:

- GEMM-like kernels are the right place to try more aggressive packaging first
- fused kernels should inherit only the lessons that survive added live state

### 3. Tune live state before chasing exotic synchronization

When a kernel is slow or wrong, prioritize:

- reducing long-lived cooperative tensors
- reducing staged footprint
- making tiles align with scale boundaries
- separating hot and cold paths

Do **not** start with:

- extra barriers
- fancy launch-order remaps
- huge tile increases

In this workspace, most durable gains came from:

- better staging policy
- lower staged footprint
- fixed reduction / scale semantics

not from synchronization tricks.

### 4. Baseline fairness matters more than people think

Several fake wins in tensor-op tuning come from benchmarking a weak baseline, not from improving the experimental path.

This happened in GEMM-style tuning too:

- int8-tuned block shapes were reused as fp16 baselines
- migrated kernels were compared on non-representative shapes
- bench selectors drifted away from production selectors

General rule:

- maintain separate baseline and experimental block-shape helpers
- keep migrated-kernel comparisons on representative shapes
- if you change a selector, update the compare bench before claiming a cliff or a win

## Threadgroup Staging Guidance

### 1. Threadgroup staging is not automatically good or bad

The correct question is:

- for this kernel, at this precision, with this footprint, is staging correctness-safe and performance-positive?

Examples from this workspace:

- Dense `NAAttention` backward:
  - staging was initially broken at low precision for larger heads
  - after reducing staged footprint, staged paths became both correct and fast
- `NAInt8Attention` backward:
  - query-side extra staging regressed and was reverted
  - key/value-side staging survived and stayed useful

So the rule is:

- stage only when the footprint and operand role make sense

### 2. Watch the threadgroup-footprint cliff

A key dense backward result was that staged low-precision `D >= 128` broke at one staged-footprint region and became correct again after lowering `executionSIMDGroups`.

For dense backward staging, the relevant footprint was:

- `headDimension * blockR * executionSIMDGroups * element_size * 2`

For fp16 staged backward with:

- `blockR = 16`
- two staged operands

the critical observation was:

- `sg = 8`, `D = 128` gave `65536` bytes and was problematic
- `sg = 6`, `D = 128` gave `49152` bytes and was safe

General rule:

- if a low-precision staged path fails first at a specific head size, compute the staged footprint
- if correctness returns after lowering `executionSIMDGroups`, treat the problem as a staged-footprint / live-state cliff, not necessarily a math bug

### 3. Query and key/value are not interchangeable

Even if two kernels look symmetric algebraically, they may not behave symmetrically under tensor-op staging.

In this workspace:

- some query-side staged ideas regressed and were reverted
- key/value-side staging remained useful and correct

So do not assume:

- “if key/value staging worked, query staging should work the same way”

Treat them as separate kernels.

### 4. Quantizers are not ordinary compute kernels

For `NAInt8MatMul`-style quantizers and related quantized attention quantizers:

- source staging in threadgroup memory was not the first win
- vectorized loads/stores were often more important
- specialized kernels for different operand roles were worth it

General rule:

- do not assume a quantizer should be optimized like a compute-heavy tensor-op kernel
- first try:
  - vectorized memory operations
  - operand-specific specialization
  - simpler direct memory paths
- only then revisit threadgroup staging

## `executionSIMDGroups` Guidance

### 1. Do not overfit a global simdgroup heuristic too early

For dense `NAAttention`, early sweeps suggested multiple contradictory “winning” simdgroup counts depending on:

- sequence length
- head count
- block shape
- thermal state

That means:

- an apparent win from `sg = 8` or `sg = 16` can be shape-specific noise

Use broad sweeps before landing a selector rule.

For GEMM-like kernels, also remember:

- simdgroup tile size and threadgroup tile size are separate knobs
- larger simdgroup tiles can regress once operand fragments no longer fit comfortably in fast thread-local state

So when a GEMM-like kernel slows down after “making the tile bigger,” check whether you enlarged:

- the traversal tile
- the threadgroup packaging
- or the simdgroup math tile

These are not interchangeable changes.

### 2. Sometimes `executionSIMDGroups` is a correctness parameter

This was the important dense backward lesson.

For low-precision staged dense backward:

- `sg = 8` was wrong once head size / staged footprint grew large enough
- `sg = 6` fixed correctness for `D >= 128`

So when staged correctness depends on `sg`, do not treat `sg` as a pure performance knob.

### 3. For int8 fused kernels, best `sg` depends on the rest of the schedule

For int8 attention experiments:

- the best `sg` changed after barrier tuning
- it changed again after Morton launch order

General rule:

- never freeze `sg` based on one schedule
- retune it after any change to:
  - barrier cadence
  - launch order
  - staging policy
  - traversal shape

## Launch Order Guidance

### 1. Morton order is not universally good

Morton launch order is a strong tool when:

- output tiles form a GEMM-like 2D grid
- locality across neighboring tiles matters

It was a real win for:

- `NAMatMul`
- `NAInt8MatMul`
- the int8 attention scaffold in bench-only tuning

It was **not** a universal win for dense backward attention. For dense `NAAttention` backward:

- full backward Morton did not hold up
- a simpler head-major key/value launch order gave a smaller, more reliable gain

General rule:

- use Morton when the launch grid really behaves like a 2D tiled output problem
- do not assume it maps cleanly onto `(sequence, head, batch)` kernels

For `NAMatMul` and `NAInt8MatMul`, Morton is easier to defend:

- the output really is a tiled 2D surface
- neighboring tiles reuse nearby input regions
- locality wins are more direct than in attention

So:

- Morton should be one of the first launch-order experiments for GEMM-like kernels
- in fused attention backward, try simpler remaps first

### 2. A simpler launch-order change can beat a fancier one

In dense backward, the useful change was:

- keep adjacent threadgroups on the same head
- sweep row groups within that head

That was smaller and more reliable than a full Morton remap.

So when in doubt:

- try simple head-major or row-major remaps before implementing a full Morton decoder

## Barrier Guidance

### 1. Extra barriers are not a default optimization

A barrier that sounds plausible can easily be a regression.

This happened repeatedly in this workspace.

### 2. `threadgroup_barrier(mem_none)` can be shape- and kernel-specific

Bench-only int8 forward experiments showed:

- periodic `threadgroup_barrier(mem_flags::mem_none)` over traversal could help on some very large shapes
- the winning cadence was not obvious
- cadence `2` became the first genuinely promising one
- the preferred simdgroup count changed once the barrier was added

But:

- dense backward barrier experiments were not worth keeping
- MLX / Steel-style barrier placement did not necessarily transfer

General rule:

- barriers are a second-order tuning knob
- only sweep them after:
  - block shape
  - staging policy
  - launch order
  are already reasonable

### 3. Barrier placement matters

If a barrier helps, its placement still matters.

In the int8 work:

- a tail barrier before the next major traversal chunk helped
- MLX-style barriers inside the current chunk were worse

So:

- do not copy barrier placement from another kernel family without measurement

## Quantization Guidance

### 1. Align quantization-scale tiles with reduction structure

This was the most important int8 backward correctness lesson.

If a reduction spans multiple quantization-scale tiles, then:

- a single post-reduction dequant factor is wrong

That is exactly what broke `NAInt8Attention` backward key/value:

- `Q` and `dO` were quantized with 16-row scales
- the backward reduction traversed 32 rows
- the kernel used one `q_scale` / `dO_scale` for the whole 32-row reduction

That caused training instability.

General rule:

- if a reduction mixes multiple scale tiles, do one of:
  - apply the correct scale per row / per sub-tile before accumulation
  - split the reduction on scale-tile boundaries
  - change the quantization tile size so the reduction and scale tile align

Do **not** accumulate mixed-scale values and dequantize once at the end unless the scale is truly constant over the reduction.

### 2. Post-reduction dequant is only valid when the scale is constant over the reduction

This is the reason the old `cP_q @ dO_q` path was structurally dangerous:

- once multiple `dO_scale` values are mixed into one `int32` accumulation, the information needed for correct dequantization is gone

General rule:

- if you want to quantize both sides of a reduction, ensure the scale tile matches the reduction extent

### 3. Diffuse softmax rows can underflow a quantized `P`

For int8 backward `dV`, the `cP_q = round(P * 127)` path can underflow badly on broad attention rows.

This is especially visible on:

- synthetic random `Q/K`
- long sequences

and less visible on:

- correlated `Q/K`
- peaky real-model attention

General rule:

- if a quantized softmax-probability path is part of your design, validate it on both:
  - broad synthetic rows
  - more realistic correlated `Q/K`

Do not mistake a diffuse-random stress test for a representative model distribution.

### 4. Specialize quantizers aggressively

Good int8 quantizer wins in this workspace came from:

- separate specialized `Q`, `K`, and `V` quantizers
- asymmetric thread counts for `Q` vs `K/V`
- `vec4` loads and stores when `D % 4 == 0`

Bad idea that was tested and rejected:

- staging the source tile in threadgroup memory just to reuse it for both max-reduction and quantization

General rule:

- first optimize quantizers with:
  - specialization
  - vectorized memory ops
  - asymmetric thread counts
- do not assume threadgroup staging will help a quantizer

These lessons came from simpler kernels too, not just attention:

- separate `A` / `B` or `Q` / `K` / `V` quantizers can beat a single generic path
- `vec4` loads/stores matter when alignment allows them
- asymmetric thread counts are often correct because different operands have different reuse and writeback costs

## Precision Guidance

### 1. Keep the algorithmic contract clear

Separate:

- IO precision
- intermediate precision
- accumulation precision

Do not conflate them.

For example, a kernel can use:

- low-precision IO
- float softmax reconstruction
- half accumulation for selected outputs

and still be the right design.

### 2. Low-precision intermediates are not automatically the main error source

In dense backward debugging:

- changing `cDS`
- changing stored `D`
- or forcing those toward float

did not explain the main failure

The real issue was staged-footprint correctness.

General rule:

- do not assume every low-precision bug is a precision bug
- often it is really:
  - footprint
  - scale alignment
  - scheduling

## Benchmarking Guidance

### 1. Build a correctness harness first

A good performance workflow has three layers:

1. direct correctness harness
2. isolated kernel bench
3. end-to-end compare bench

Why:

- the direct harness prevents optimizing invalid kernels
- the isolated bench identifies the real cause of a win or cliff
- the compare bench tells you whether the change matters end to end

### 2. Separate selector cliffs from kernel cliffs

If a performance cliff appears at one shape:

- force both candidate paths in the isolated bench
- see whether the cliff is from:
  - the selector
  - or the kernel itself

This was essential in the dense backward `3072` investigation.

### 3. Run thermal-sensitive sweeps carefully

For long-sequence Apple GPU benchmarks:

- run one shape per process
- run larger cases first
- insert cooldown gaps
- repeat one anchor shape at the end

Use:

- `best3_avg_ms`

as the primary cross-case metric.

### 4. Keep baselines fair

Do not reuse an experimental block shape as the baseline if it penalizes the baseline path.

This mistake can create fake speedups.

Maintain:

- separate baseline block-shape helpers
- separate experimental block-shape helpers

### 5. For migrated kernels, performance parity needs representative-shape checks

A migration that preserves correctness can still hide a performance regression if benchmark shapes are too narrow.

For GEMM-like and attention-like kernels alike:

- benchmark the real production anchors
- include at least one smaller shape, one medium shape, and one large shape
- if a kernel family uses batching over `threadgroup_position_in_grid.z`, include representative batched shapes too

Do not claim a win from one anchor if the migration target serves a broader shape family.

## Convolution / Conv3D Lessons

`NAConv3D` is not the main focus of this skill, but it produced a few reusable tensor-op lessons that apply beyond convolution.

### 1. Keep backend-selection decisions separate

If a frontend can route work to multiple accelerated paths, do not collapse them into one boolean just because the kernels are related.

This mattered for Conv3D:

- `use_mfa_gemm` and `use_mfa_conv3d` needed to stay separate

General rule:

- keep backend-selection logic aligned to the actual kernel family
- do not let a generalized frontend switch hide the real support surface of a narrower kernel

### 2. Batch in the grid when the math factorizes cleanly

One of the most reusable `NAConv3D` lessons was the same pattern used by `NAMatMul`:

- do not loop batch on the host if the kernel can naturally encode batch in `threadgroup_position_in_grid.z`

This is worth calling out because it often improves both code structure and performance:

- host dispatch stays simpler
- the kernel owns the full batching pattern
- launch geometry stays closer to the true mathematical decomposition

General rule:

- if a kernel naturally decomposes by batch or depth slice, try encoding that in grid `z` before introducing host-side outer loops

### 3. Treat layout normalization as part of the kernel design, not a side detail

For `NAConv3D`, weights arrived in OIDHW / NCHW-style layout but the MFA kernel wanted DHWIO-style scratch.

The lesson is broader than Conv3D:

- layout conversion and packing are part of the performance design
- if the kernel consumes a special packed layout, plan for that explicitly in scratch sizing, caching, and validation

General rule:

- do not bury layout normalization as an afterthought
- make it explicit in the dataflow and scratch plan

### 4. Scratch planning should reserve non-overlapping regions up front

`NAConv3D` mirrored GEMM-style scratch reservation:

- reserve the front of scratch for the permuted weights
- place later scratch users after that reserved region

This is a general tensor-op lesson:

- scratch consumers should not opportunistically overlap unless the lifetime proof is obvious
- reserve major regions intentionally

General rule:

- design scratch layout the same way you design tile layout
- if one phase depends on a transformed or depalettized view, reserve that space first and build later scratch allocations around it

### 5. Descriptor / kernel-descriptor coherence is a correctness issue

`NAConv3D` reinforced a rule that applies to all generated MFA kernels:

- every non-derived value that changes generated source must live in the descriptor that keys the kernel

Otherwise:

- cached source
- runtime params
- and dispatch logic

can silently diverge.

General rule:

- if a property affects code generation, put it in the kernel descriptor
- if it is a runtime-only dispatch value, keep it out of the source cache key
- do not split one source-generation fact across descriptor and `setBytes`

### 6. Narrow support surfaces are a feature, not a bug

`NAConv3D` only supported a deliberately narrow shape family at first.

That was the right move.

General rule:

- ship a narrow, explicit support surface before generalizing
- document the constraints directly
- only widen after correctness, scratch, and dispatch assumptions are stable

## Code Organization Guidance

### 1. Remove diagnostic branches once the winning path is clear

During tuning, it is fine to add:

- bench-only modes
- alternate schedules
- barrier knobs
- launch-order knobs

But once the winner is clear:

- remove dead or compile-time-false branches
- keep only the modes that still bound the current design

This keeps generator complexity under control.

### 2. Keep cache keys and runtime shape responsibilities separate

For MFA kernels:

- kernel-object cache should key source-generation properties
- runtime shape should drive dispatch and function constants at encode time

Do not store shape-dependent dispatch assumptions in a reused kernel object unless that is truly part of source generation.

## Attention-Specific Rules Of Thumb

For this workspace, the most robust starting points were:

- Dense `NAAttention`
  - `blockR = 16`
  - `blockC = 64`
  - staged low-precision backward
  - reduced `executionSIMDGroups` for larger heads when staged footprint demands it

- `NAInt8Attention`
  - treat reduction / scale alignment as a first-class design constraint
  - keep query-side extra staging conservative
  - key/value-side staging is more promising
  - benchmark realistic correlated `Q/K` if evaluating quantized `P` behavior

## GEMM-Like Rules Of Thumb

For this workspace, the most reusable `NAMatMul` / `NAInt8MatMul` lessons were:

- Morton launch order is a first-class tuning knob for tiled 2D outputs
- do not conflate simdgroup math-tile size with threadgroup packaging size
- larger simdgroup tiles can regress even when larger threadgroup reuse still helps
- int8 GEMM can be much faster than fp16 GEMM, but that speedup is not a promise for fused kernels
- specialize quantizers instead of over-generalizing them
- prefer vectorized quantizer memory ops before trying threadgroup staging
- keep baseline and experiment scheduling helpers separate so benchmark claims stay honest

## Checklist For Future Agents

Before changing a tensor-op kernel:

1. Identify whether the problem is:
   - correctness
   - selector
   - performance
2. Build or reuse a direct correctness harness.
3. Confirm whether the benchmark selector matches production.
4. Measure forced-path behavior in an isolated bench.
5. Only then run end-to-end compare benchmarks.
6. If a staged low-precision path is broken:
   - compute staged footprint
   - check whether lowering `executionSIMDGroups` fixes correctness
7. If a quantized reduction is wrong:
   - check whether the reduction spans multiple scale tiles
8. If a launch-order tweak is proposed:
   - compare it against a simple head-major remap before implementing full Morton
9. If a barrier tweak is proposed:
   - treat it as bench-only until it shows a repeatable win

## Short Version

If you only remember one page from this skill, remember this:

- correctness first
- `blockR = 16`
- bigger traversal is usually not the answer for fused attention
- staged footprint can be a correctness cliff
- reduction / scale alignment is a real correctness issue in quantized kernels
- Morton can help, but not every `(sequence, head, batch)` kernel is a GEMM
- extra barriers are not free and should be treated as experiments
- benchmark selectors can lie if they drift from production

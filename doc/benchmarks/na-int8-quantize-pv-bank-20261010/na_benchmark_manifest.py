#!/usr/bin/env python3
"""Deterministic application and selector-boundary workloads (no external files)."""
import json

DRAW_THINGS_REVISION = 'afed99ce5981875e3b3aa7d19059a516c73c5f2f'


def manifest():
    cases = {}

    def add(name, op, shape, tags, *, flags=0, causal=False, cast=False,
            variants=None, source='', description=''):
        key = (op, tuple(shape), flags, causal, cast)
        if variants is None:
            variants = ['fp16', 'fp32', 'int8', 'mlx'] if op == 'matmul' else ['fp16', 'int8', 'mlx']
        if key in cases:
            c = cases[key]
            c['tags'] = sorted(set(c['tags'] + tags))
            c['aliases'].append(name)
            c['variants'] = list(dict.fromkeys(c['variants'] + variants))
            return
        cases[key] = dict(id=name, aliases=[], operation=op, shape=shape,
                          dispatch_flags=flags, causal=causal, output_cast=cast,
                          variants=variants, tags=sorted(set(tags)),
                          source=source, description=description)

    models = [
        ('flux2', 6144, 18432, 48, 48, 128, 512, 'Flux2.swift'),
        ('krea2', 6144, 16384, 48, 12, 128, 256, 'Krea2.swift'),
        ('ideogram4', 4608, 12288, 18, 18, 256, 256, 'Ideogram4.swift'),
        ('qwen21', 4096, 12288, 32, 32, 128, 256, 'QwenImage2_1.swift'),
    ]
    for model, channels, ffn, hq, hk, d, text, file in models:
        source = 'Libraries/SwiftDiffusion/Sources/Models/' + file
        for resolution in [512, 768, 1024, 1536, 2048]:
            image = (resolution // 16) ** 2
            r = image if model == 'qwen21' else image + text
            add(f'{model}-{resolution}-attention', 'attention', [r, image+text, d, 1, hq, hk],
                ['inference'] + (['smoke'] if resolution == 512 else []), source=source,
                description=f'{resolution}px square; {text} assumed text tokens; exact dense SDPA')
            rows = image if model in ['flux2', 'qwen21'] else image + text
            for role, n, k in [('projection', channels, channels), ('ffn-up', ffn, channels), ('ffn-down', channels, ffn)]:
                add(f'{model}-{resolution}-{role}', 'matmul', [rows, n, k, 1, 1, 1], ['inference'], source=source,
                    description='Image stream for FLUX/Qwen; joint stream for Krea/Ideogram')
        add(f'{model}-last-image-only', 'attention', [4096,4096+text,d,1,hq,hk], ['inference'], source=source)
        add(f'{model}-reference', 'attention', [4096,8192+text,d,1,hq,hk], ['inference'], source=source,
            description='One image-reference token block; exact dense SDPA')
        add(f'{model}-1024-vjp', 'backward', [4096 if model=='qwen21' else 4096+text,4096+text,d,1,hq,hk],
            ['backward'], source=source, description='Saved forward excluded from both CCV and actual MLX VJP timing')
    source = ('Libraries/SwiftDiffusion/Sources/Models/MiniMaxH3.swift:6,166,194,279,740; '
              'Libraries/SwiftDiffusion/Sources/Models/UNetProtocol.swift:767,819; '
              'Libraries/LocalImageGenerator/Sources/LocalImageGenerator.swift:4265')
    # 240 requested frames round to 72 latent / 243 decoded frames. Spatial
    # compression is 16, then H3 makes 2x2 latent patches. Joint attention also
    # includes 256 assumed text tokens and 810 audio tokens, with no references.
    for label, width, height in [('480p-ui-10s',832,512), ('720p-ui-10s',1280,768)]:
        rows = 72 * (height // 32) * (width // 32) + 256 + 810
        add('h3-'+label, 'attention', [rows,rows,128,1,56,56], ['inference','long'], source=source,
            description='832x512 or 1280x768 UI canvas, 10 seconds requested / 243 frames; 256 text + 810 audio tokens, no references; dense SDPA, not Sol')
        for role,n,k in [('q',7168,5376), ('output',5376,7168), ('ffn-up',14336,5376), ('ffn-down',5376,14336)]:
            add(f'h3-{label}-{role}', 'matmul', [rows,n,k,1,1,1], ['inference','long'], source=source)
            if role in ['output','ffn-down']:
                add(f'h3-{label}-{role}-cast', 'matmul', [rows,n,k,1,1,1], ['inference','long'], cast=True, source=source,
                    description='Includes half-rounded Float32 output; INT8 requests fused store with normal fallback')
    add('h3-short-vjp', 'backward', [8948,8948,128,1,56,56], ['backward','long'], source=source)
    # Local Code prefill projections. Dimensions come from Qwen3.5 configurations.
    source = 'Libraries/SwiftLLM/Sources/Models/Qwen3_5.swift:60,69,77,299; Apps/LocalCode/Sources/Models/Qwen3_5TextGenerator.swift:338,407,692'
    for model,channels,ffn,heads in [('4b',2560,9216,16),('9b',4096,12288,16),('27b',5120,17408,24)]:
        for rows in [1,2,8,32,128,512,1024,2048,3072,4096]:
            if channels != heads*256:
                add(f'gemm-square-control-{channels}-m{rows}', 'matmul', [rows,channels,channels,1,1,1], ['coverage'], flags=1,
                    description='Synthetic hidden-size square, not a Qwen attention projection')
            for role,n,k in [('q',heads*256,channels),('kv',1024,channels),('output',channels,heads*256),
                             ('up',ffn,channels),('down',channels,ffn)]:
                add(f'local-qwen35-{model}-m{rows}-{role}', 'matmul', [rows,n,k,1,1,1], ['inference'], flags=1, source=source,
                    description='Dynamic-M prefill/decode; dense half or rowwise INT8 weights already expanded')
    for rows in [1,8,128,512,2048]:
        for cols in [2048,8192]:
            add(f'local-gqa-r{rows}-c{cols}', 'attention', [rows,cols,256,1,16,4], ['inference'], flags=3, causal=True,
                source=source, description='Causal growing-KV-cache geometry, dynamic R/C')
    for rows in [1,128,2048]:
        for cols in [2048,8192]:
            add(f'local-27b-gqa-r{rows}-c{cols}', 'attention', [rows,cols,256,1,24,4], ['inference'], flags=3, causal=True,
                source=source, description='Qwen3.5 27B causal growing cache')
    # Coherent geometric neighborhoods: row minima, output area, N/K alignment,
    # register/native crossover, deep reductions and partial fragments.
    for rows in [511,512,513,1024,1536,2047,2048,2049,3072,4095,4096,4097,8191,8192,8193]:
        for n,k in [(1536,4096),(6144,6144),(6144,18432),(12288,8192)]:
            add(f'gemm-rows-{rows}-{n}-{k}', 'matmul', [rows,n,k,1,1,1], ['coverage'])
    for n in [1535,1536,1537,4095,4096,4097]:
        for k in [2047,2048,2049,8191,8192,8193,16383,16384,16385,32767,32768,32769]:
            add(f'gemm-edges-n{n}-k{k}', 'matmul', [4097,n,k,1,1,1], ['coverage'], variants=['fp32','int8','mlx'])
    for rows in [2048,3072,4096,6144,8192]:
        for n,k in [(2048,8192),(2560,9216),(4096,12288),(6144,16384),(8192,24576)]:
            add(f'gemm-aspect-{rows}-{n}-{k}', 'matmul', [rows,n,k,1,1,1], ['coverage'])
    for rows in [2048,4096,8192]:
        add(f'gemm-dynamic-tail-{rows}', 'matmul', [rows,6145,9217,1,1,1], ['coverage'], flags=1, cast=True)
    # Keep known native low-precision failures visible and separate from passing baselines.
    for n in [12287,12289]:
        add(f'gemm-lowp-stress-n{n}', 'matmul', [512,n,24576,1,1,1], ['stress'], variants=['fp16','fp32','int8','mlx'],
            description='Known existing native half-accumulator CPU-L2 error >1%; failure must not become a speedup')
    for d in [64,128,256]:
        for r,c in [(16,65),(64,64),(65,128),(255,257),(256,257),(257,257),(1023,1025),(2048,4095),
                    (4095,4096),(4096,4096),(4097,4097),(4096,8191),(4096,8192),(4096,8193),(8193,4097)]:
            add(f'attn-d{d}-r{r}-c{c}', 'attention', [r,c,d,1,8,8], ['coverage']+(['smoke'] if (r,c)==(257,257) else []))
        for b,hq,hk in [(2,8,8),(1,12,3),(1,18,6),(1,56,7)]:
            add(f'attn-groups-d{d}-b{b}-h{hq}-{hk}', 'attention', [1025,2051,d,b,hq,hk], ['coverage'])
        add(f'attn-dynamic-d{d}', 'attention', [4097,8193,d,1,8,2], ['coverage'], flags=3)
        add(f'attn-causal-d{d}', 'attention', [257,2049,d,1,8,2], ['coverage'], causal=True, flags=3)
        add(f'vjp-d{d}', 'backward', [513,1025,d,1,8,2], ['backward','smoke'])
    for r,c,h in [(4095,24576,48),(4096,24575,48),(4096,24576,48),(4097,24577,48),
                  (4096,32767,32),(4096,32768,32),(4096,32769,32),(8193,24576,56)]:
        add(f'attn-partition-r{r}-c{c}-h{h}', 'attention', [r,c,128,1,h,h], ['coverage'])
    # D256 causal register inference: selector edge, nonaligned KV prefix,
    # complete/partial query tiles, batches and grouped query heads.
    for r,c,b,hq,hk in [(r,257,1,8,2) for r in [64,65,127,128,255,256,257]] + [
            (65,65,1,8,2),(257,513,1,8,2),(513,1025,2,6,2),
            (4096,8192,1,24,4),(4097,8193,1,8,2)]:
        add(f'causal-register-r{r}-c{c}-b{b}-h{hq}-{hk}', 'attention', [r,c,256,b,hq,hk],
            ['coverage'], flags=3, causal=True,
            description='D256 causal inference boundary; right-aligned KV prefix, dynamic lengths; INT8 request can fall back to FP16')
    for rows in [512,2048,4096]:
        add(f'smoke-gemm-m{rows}', 'matmul', [rows,4096,4096,1,1,1], ['smoke','coverage'])
    return dict(version=3, draw_things_revision=DRAW_THINGS_REVISION,
                assumptions='Source-derived dimensions; selected image sizes/prompt lengths. FP16 IO; weights preexpanded; INT8 includes activation quantization. GPU operator timing, not whole generation. No attention storage transpose.',
                workloads=list(cases.values()))


if __name__ == '__main__':
    print(json.dumps(manifest(), indent=2))

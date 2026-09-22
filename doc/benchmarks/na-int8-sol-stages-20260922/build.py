#!/usr/bin/env python3
"""Build isolated comparison kernels without changing production sources."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

here = Path(__file__).resolve().parent
repo = here.parents[2]
output = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="sol-stages-"))
output.mkdir(parents=True, exist_ok=True)
files = ["ccv_nnc_mfa_sol_attention.cpp"] + [
    "kernels/NAInt8SolAttention" + suffix
    for suffix in ["Descriptor.hpp", "Descriptor.cpp", "Kernel.hpp", "Kernel.cpp"]
]
expected = json.loads((here / "base-sha256.json").read_text())
for relative in files:
    actual = hashlib.sha256((repo / "lib/nnc/mfa" / relative).read_bytes()).hexdigest()
    if actual != expected[relative]:
        raise SystemExit("Experiment base differs: " + relative + "; use the revision containing these experiments.")
extra = []
for variant, name in [("two-pass", "SolBaselineAttention"), ("fusion", "SolExperimentAttention"), ("seven-launch", "SolSeparateAttention")]:
    target = output / variant
    (target / "kernels").mkdir(parents=True, exist_ok=True)
    for relative in files:
        shutil.copyfile(repo / "lib/nnc/mfa" / relative, target / relative)
    subprocess.run(["patch", "--batch", "-p1", "-d", str(target), "-i", str(here / "seven-launch.patch")], check=True)
    if variant != "seven-launch":
        subprocess.run(["patch", "--batch", "-p1", "-d", str(target), "-i", str(here / (variant + ".patch"))], check=True)
    for relative in files:
        path = target / relative
        text = path.read_text().replace("NAInt8SolAttention", name).replace("NAINT8SOLATTENTION", name.upper())
        if variant == "seven-launch":
            text = text.replace("ccv_nnc_mfa_encode_sol_attention(", "encode_sol_separate(")
        renamed = target / relative.replace("NAInt8SolAttention", name)
        renamed.write_text(text)
        if renamed != path:
            path.unlink()
        if renamed.suffix == ".cpp":
            extra.append(str(renamed))
flags = ["clang++", "-std=c++17", "-O3", "-fblocks", "-Wno-deprecated-declarations"]
for directory in ["lib", "lib/nnc/mfa", "lib/nnc/mfa/kernels"]:
    flags += ["-I" + str(repo / directory)]
link = [str(repo / "lib/libccv.a")]
for framework in ["Accelerate", "Metal", "Foundation", "QuartzCore", "MetalPerformanceShaders", "MetalPerformanceShadersGraph", "CoreML", "CoreVideo", "IOSurface"]:
    link += ["-framework", framework]
subprocess.run(flags + [str(here / "compare.cpp")] + extra + link + ["-o", str(output / "compare")], check=True)
subprocess.run(flags + [str(here / "mean_verify.cpp")] + link + ["-o", str(output / "mean_verify")], check=True)
subprocess.run(flags + ["-I" + str(output / "fusion/kernels"), str(here / "resources.cpp")] + [path for path in extra if "/fusion/" in path] + link + ["-o", str(output / "resources")], check=True)
print("Resources:", output / "resources")
print("Comparison:", output / "compare")
print("Mean verification:", output / "mean_verify")

"""Build the floating-PV prototype probe for macOS or a separate signed iPad app."""
import argparse
import concurrent.futures
import json
import re
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ios", action="store_true")
parser.add_argument("--large-memory", action="store_true", help="Request the supported iPad increased-memory limit for H3 workloads")
parser.add_argument("--kernel-source", type=Path, help="Experimental source snapshot; leave repository files untouched")
parser.add_argument("--probe-source", type=Path, help="Experimental probe source")
parser.add_argument("--bundle", default="com.liu.NAFloatPVAttentionExperiment")
parser.add_argument("--team", help="Override the development team in the project template")
args = parser.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[2]
template = here.parent / "na-int8-attention-hadamard-cross-device-20261007/ios"
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
lib = repo / "lib"
mfa = lib / "nnc/mfa"
shutil.copyfile(args.probe_source or repo / "bin/mfa/na_int8_attention_float_pv_bench.cpp", out / "probe.cpp")
shutil.copyfile(here / "NAInt8TuningApp.mm", out / "NAInt8TuningApp.mm")
project = out / "NAInt8TuningApp.xcodeproj"
project.mkdir(exist_ok=True)
text = (template / "project.pbxproj").read_text()
text = text.replace("/private/tmp/ccv-attn-hadamard-ios-20261007", str(out))
text = re.sub(r"/Users/[^/]+/workspace/ccv", lambda _: str(repo), text)
text = text.replace("CCV_NA_HADAMARD_EMBEDDED", "CCV_NA_PV_EMBEDDED")
text = text.replace("com.liu.NAHadamardAttentionExperiment", args.bundle)
if args.team:
    text = text.replace("6HP3U6Z7P6", args.team)
(project / "project.pbxproj").write_text(text)

# These GPU probes never use the ANE rowwise cache or command-dispatch flags.
(out / "support.cpp").write_text(
    '#include "nnc/mfa/ccv_nnc_mfa.hpp"\n'
    'extern "C" uint64_t ccv_nnc_flags(void) { return 0; }\n'
    'void ccv_nnc_mfa_ane_rowwise_gemm_cleanup(ccv_nnc_mfa_context_t*) {}\n'
)
files = [mfa / name for name in [
    "Metal.cpp", "ccv_nnc_mfa.cpp", "ccv_nnc_mfa_error.cpp",
    "3rdparty/metal-cpp/Dispatch.cpp", "kernels/CodeWriter.cpp",
    "kernels/NAInt8AttentionDescriptor.cpp", "kernels/NAInt8AttentionKernelDescriptor.cpp",
    "kernels/NAInt8AttentionKernel.cpp", "kernels/NAAttentionDescriptor.cpp",
    "kernels/NAAttentionKernelDescriptor.cpp", "kernels/NAAttentionKernel.cpp",
    "kernels/ANERowwiseTransformDescriptor.cpp",
    "kernels/ANERowwiseTransformKernelDescriptor.cpp", "kernels/ANERowwiseTransformKernel.cpp",
    "kernels/GEMMHeaders.cpp",
]] + [out / "support.cpp", lib / "ccv_util.c"]
if args.kernel_source:
    snapshot = out / "kernels"
    snapshot.mkdir(exist_ok=True)
    for source in args.kernel_source.glob("NAInt8Attention*.*"):
        if source.suffix in (".hpp", ".cpp"):
            (snapshot / source.name).write_text(source.read_text().replace('"../', '"' + str(mfa) + '/'))
    files = [snapshot / source.name if source.name.startswith("NAInt8Attention") else source for source in files]
    probe_text = (out / "probe.cpp").read_text()
    for source in snapshot.glob("*.hpp"):
        probe_text = probe_text.replace('"nnc/mfa/kernels/'+source.name+'"', '"'+str(source)+'"')
    (out / "probe.cpp").write_text(probe_text)
sdk = subprocess.check_output(["xcrun", "--sdk", "iphoneos" if args.ios else "macosx", "--show-sdk-path"], text=True).strip()

def compile_source(source):
    obj = out / (source.name + ".o")
    command = ["xcrun", "clang" if source.suffix == ".c" else "clang++"]
    if source.suffix != ".c":
        command += ["-std=gnu++17"]
    command += ["-target", "arm64-apple-ios26.0" if args.ios else "arm64-apple-macos26.0", "-isysroot", sdk, "-O3", "-fblocks", "-w",
                "-I" + str(lib), "-I" + str(mfa / "kernels"), "-DHAVE_MPS", "-DHAVE_ACCELERATE_FRAMEWORK",
                "-c", str(source), "-o", str(obj)]
    with (out / (source.name + ".build.log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return str(obj), command

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(compile_source, files))
archive = ["xcrun", "ar", "rcs", str(out / "libmfa.a"), *[obj for obj, _ in results]]
subprocess.run(archive, check=True)
if args.ios:
    build = ["xcodebuild", "-project", str(project), "-scheme", "NAInt8TuningApp",
             "-configuration", "Release", "-sdk", "iphoneos", "-derivedDataPath", str(out / "derived"),
             "-allowProvisioningUpdates", "build"]
    if args.large_memory:
        import plistlib
        entitlements = out / "Benchmark.entitlements"
        entitlements.write_bytes(plistlib.dumps({"com.apple.developer.kernel.increased-memory-limit": True, "com.apple.developer.kernel.extended-virtual-addressing": True}))
        build += ["CODE_SIGN_ENTITLEMENTS=" + str(entitlements)]
    product = out / "derived/Build/Products/Release-iphoneos/NAInt8TuningApp.app"
else:
    build = ["xcrun", "clang++", "-std=c++17", "-O3", "-fblocks", "-w", "-Wl,-dead_strip", "-DHAVE_MPS",
             "-I" + str(lib), str(out / "probe.cpp"), str(out / "libmfa.a"),
             "-framework", "Foundation", "-framework", "Metal", "-framework", "QuartzCore",
             "-framework", "Accelerate", "-framework", "MetalPerformanceShaders", "-framework", "IOKit",
             "-o", str(out / "probe")]
    product = out / "probe"
(out / "commands.json").write_text(json.dumps([cmd for _, cmd in results] + [archive, build], indent=2))
with (out / "build.log").open("w") as log:
    subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
import hashlib
metadata = {str(source.relative_to(repo)): hashlib.sha256(source.read_bytes()).hexdigest()
            for source in files if source.is_relative_to(repo)}
if args.kernel_source:
    metadata.update({"experimental/"+source.name: hashlib.sha256(source.read_bytes()).hexdigest() for source in (out/"kernels").glob("*")})
metadata["bin/mfa/na_int8_attention_float_pv_bench.cpp"] = hashlib.sha256((out / "probe.cpp").read_bytes()).hexdigest()
(out / "sources.json").write_text(json.dumps(metadata, indent=2))
print(product)

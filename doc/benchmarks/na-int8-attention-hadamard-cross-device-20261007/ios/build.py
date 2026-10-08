"""Build the production-kernel probe as a separate signed iPad app."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
parser.add_argument("--team", help="Override the development team in the project template")
args = parser.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[3]
out = (args.output or Path(tempfile.mkdtemp(prefix="ccv-attention-hadamard-ios-"))).resolve()
out.mkdir(parents=True, exist_ok=True)
lib = repo / "lib"
mfa = lib / "nnc/mfa"
shutil.copyfile(repo / "bin/mfa/na_int8_attention_hadamard_bench.cpp", out / "probe.cpp")
shutil.copyfile(here / "NAInt8TuningApp.mm", out / "NAInt8TuningApp.mm")
project = out / "NAInt8TuningApp.xcodeproj"
project.mkdir(exist_ok=True)
text = (here / "project.pbxproj").read_text()
text = text.replace("/private/tmp/ccv-attn-hadamard-ios-20261007", str(out))
text = text.replace("/Users/liu/workspace/ccv", str(repo))
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
    "kernels/NAInt8AttentionKernel.cpp", "kernels/ANERowwiseTransformDescriptor.cpp",
    "kernels/ANERowwiseTransformKernelDescriptor.cpp", "kernels/ANERowwiseTransformKernel.cpp",
    "kernels/GEMMHeaders.cpp",
]] + [out / "support.cpp", lib / "ccv_util.c"]
sdk = subprocess.check_output(["xcrun", "--sdk", "iphoneos", "--show-sdk-path"], text=True).strip()

def compile_source(source):
    obj = out / (source.name + ".o")
    command = ["xcrun", "clang" if source.suffix == ".c" else "clang++"]
    if source.suffix != ".c":
        command += ["-std=gnu++17"]
    command += ["-target", "arm64-apple-ios26.0", "-isysroot", sdk, "-O3", "-fblocks", "-w",
                "-I" + str(lib), "-DHAVE_MPS", "-DHAVE_ACCELERATE_FRAMEWORK",
                "-c", str(source), "-o", str(obj)]
    with (out / (source.name + ".build.log")).open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return str(obj), command

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    results = list(executor.map(compile_source, files))
archive = ["xcrun", "ar", "rcs", str(out / "libmfa.a"), *[obj for obj, _ in results]]
subprocess.run(archive, check=True)
build = ["xcodebuild", "-project", str(project), "-scheme", "NAInt8TuningApp",
         "-configuration", "Release", "-sdk", "iphoneos", "-derivedDataPath", str(out / "derived"),
         "-allowProvisioningUpdates", "build"]
(out / "commands.json").write_text(json.dumps([cmd for _, cmd in results] + [archive, build], indent=2))
with (out / "xcode-build.log").open("w") as log:
    subprocess.run(build, stdout=log, stderr=subprocess.STDOUT, check=True)
print(out / "derived/Build/Products/Release-iphoneos/NAInt8TuningApp.app")

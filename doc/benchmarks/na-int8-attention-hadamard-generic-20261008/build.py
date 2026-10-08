"""Build a direct old/new GPU comparator using the recorded pre-refactor source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[2]
out = (args.output or Path(tempfile.mkdtemp(prefix="ccv-hadamard-bit-identity-"))).resolve()
out.mkdir(parents=True, exist_ok=True)
metadata = json.loads((here / "verification.json").read_text())
kernels = repo / "lib/nnc/mfa/kernels"
current = kernels / "NAInt8AttentionKernel.cpp"
old = out / "old.cpp"
subprocess.run(["patch", "-R", "-o", str(old), str(current)],
               input=(here / "old-to-generic.patch").read_bytes(), check=True)
assert hashlib.sha256(old.read_bytes()).hexdigest() == metadata["old_kernel_sha256"]

# Give the old generator separate C++ symbols; its generated Metal is unchanged.
header = (kernels / "NAInt8AttentionKernel.hpp").read_text()
header = header.replace("NAInt8AttentionKernel_hpp", "LegacyNAInt8AttentionKernel_hpp")
header = re.sub(r"\bNAInt8AttentionKernel\b", "LegacyNAInt8AttentionKernel", header)
(out / "LegacyNAInt8AttentionKernel.hpp").write_text(header)
legacy = re.sub(r"\bNAInt8AttentionKernel\b", "LegacyNAInt8AttentionKernel", old.read_text())
(out / "LegacyNAInt8AttentionKernel.cpp").write_text(legacy)

probe = (here / "compare.cpp").read_text()
probe = probe.replace('#include <iterator>', '#include <iterator>\n#include "LegacyNAInt8AttentionKernel.hpp"')
probe = probe.replace("  if (!old_source_path) return 2;\n", "")
start = probe.index("      std::ifstream input(old_source_path);")
end = probe.index("      old_cache.emplace", start)
loaded_source = probe[start:end]
probe = probe[:start] + "      if (old_source_path) {\n" + loaded_source + "      } else {\n" + (
    "        LegacyNAInt8AttentionKernel legacy(kernel_descriptor, device.get());\n"
    "        kernel->source = legacy.source;\n"
    "        kernel->library = legacy.library;\n"
    "      }\n"
    "      if (const char* path = getenv(\"CCV_NA_EXPORT_LEGACY_SOURCE\")) {\n"
    "        std::ofstream output(path); output << kernel->source;\n"
    "        if (!output.good()) return 5;\n"
    "      }\n"
) + probe[end:]
(out / "compare.cpp").write_text(probe)

base = ["xcrun", "clang++", "-std=gnu++17", "-O2", "-fblocks", "-DHAVE_MPS",
        "-I" + str(repo / "lib"), "-I" + str(kernels)]
compile_command = base + ["-c", str(out / "LegacyNAInt8AttentionKernel.cpp"),
                          "-o", str(out / "legacy.o")]
frameworks = ["Accelerate", "MetalPerformanceShaders", "MetalPerformanceShadersGraph",
              "Foundation", "CoreVideo", "CoreML", "IOSurface", "Metal", "IOKit", "QuartzCore"]
link_command = base + [str(out / "compare.cpp"), str(out / "legacy.o"),
                       str(repo / "lib/libccv.a"), "-o", str(out / "hadamard_compare"),
                       "-L/usr/local/lib", "-lm", "-lblas", "-lpthread", "-lsqlite3", "-lz"]
for framework in frameworks:
    link_command += ["-framework", framework]
with (out / "build.log").open("w") as log:
    for command in [compile_command, link_command]:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
(out / "commands.json").write_text(json.dumps([compile_command, link_command], indent=2))
print(out / "hadamard_compare")

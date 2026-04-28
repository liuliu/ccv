#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: bin/nnc/precompile_mfa_kernels.sh [options]

Options:
  --check              Generate into a temporary directory and compare with tracked outputs.
  --skip-compile       Reuse existing staged .metallib files; only regenerate selectors and .inc files.
  --stage-dir DIR      Directory for emitted .metal/.metallib files.

Environment:
  XCODE_APP            Xcode app used for Metal compilation. Defaults to /Applications/Xcode 16.4.app.
  BUILD_DIR            Directory for temporary generator binaries. Defaults to <stage-dir>/.build.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE_DIR="$ROOT/bin/nnc/split_files"
STAGE_DIR_SET=0
CHECK=0
SKIP_COMPILE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      CHECK=1
      shift
      ;;
    --skip-compile)
      SKIP_COMPILE=1
      shift
      ;;
    --stage-dir)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      STAGE_DIR="$2"
      STAGE_DIR_SET=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

XCODE_APP="${XCODE_APP:-/Applications/Xcode 16.4.app}"
DEVELOPER_DIR_16_4="$XCODE_APP/Contents/Developer"
if [[ ! -d "$DEVELOPER_DIR_16_4" ]]; then
  echo "missing Xcode developer directory: $DEVELOPER_DIR_16_4" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mfa-precompiled.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

if [[ "$CHECK" -eq 1 && "$SKIP_COMPILE" -eq 0 && "$STAGE_DIR_SET" -eq 0 ]]; then
  STAGE_DIR="$WORK_DIR/split_files"
fi

ATTENTION_DIR="$STAGE_DIR"
GEMM_DIR="$STAGE_DIR/gemm"
BUILD_DIR="${BUILD_DIR:-$STAGE_DIR/.build}"

ATTENTION_SELECTOR="$ROOT/lib/nnc/mfa/kernels/AttentionKernel+Precompiled.cpp"
GEMM_SELECTOR="$ROOT/lib/nnc/mfa/kernels/GEMMKernel+Precompiled.cpp"
ATTENTION_INC="$ROOT/lib/nnc/mfa/kernels/AttentionKernel+Precompiled.inc"
GEMM_INC="$ROOT/lib/nnc/mfa/kernels/GEMMKernel+Precompiled.inc"
PACKAGER="$ROOT/lib/nnc/mfa/packager.py"

build_generators() {
  local cxx="${CXX:-clang++}"
  mkdir -p "$BUILD_DIR"
  make -C "$ROOT/lib"
  local common_flags=(
    -std=c++17
    -O3
    -Wall
    -fblocks
    -DHAVE_CBLAS
    -DHAVE_PTHREAD
    -DHAVE_ACCELERATE_FRAMEWORK
    -DUSE_DISPATCH
    -DHAVE_MPS
    -I"$ROOT/lib"
    -I/usr/local/include
    -I/opt/homebrew/include
  )
  local link_flags=(
    "$ROOT/lib/libccv.a"
    -L/usr/local/lib
    -L/opt/homebrew/lib
    -lm
    -lblas
    -lpthread
    -framework Accelerate
    -framework MetalPerformanceShaders
    -framework MetalPerformanceShadersGraph
    -framework Foundation
    -framework CoreVideo
    -framework CoreML
    -framework IOSurface
    -framework Metal
    -lc++
  )
  "$cxx" "${common_flags[@]}" "$ROOT/bin/nnc/attention_kernel_gen.cpp" "${link_flags[@]}" -o "$BUILD_DIR/attention_kernel_gen"
  "$cxx" "${common_flags[@]}" "$ROOT/bin/nnc/gemm_kernel_gen.cpp" "${link_flags[@]}" -o "$BUILD_DIR/gemm_kernel_gen"
}

clean_kernel_dir() {
  local dir="$1"
  mkdir -p "$dir"
  find "$dir" -maxdepth 1 -type f \( -name '*.metal' -o -name '*.metallib' \) -delete
}

emit_sources() {
  clean_kernel_dir "$ATTENTION_DIR"
  clean_kernel_dir "$GEMM_DIR"
  "$BUILD_DIR/attention_kernel_gen" --emit-metal "$ATTENTION_DIR" > /dev/null
  "$BUILD_DIR/gemm_kernel_gen" --emit-metal "$GEMM_DIR" > /dev/null
}

compile_metals() {
  local dir="$1"
  local count=0
  local module_cache="$BUILD_DIR/clang-module-cache"
  mkdir -p "$module_cache"
  while IFS= read -r metal_file; do
    local stem="${metal_file%.metal}"
    DEVELOPER_DIR="$DEVELOPER_DIR_16_4" xcrun -sdk macosx metal -fmodules-cache-path="$module_cache" "$metal_file" -o "${stem}_macosx.metallib"
    DEVELOPER_DIR="$DEVELOPER_DIR_16_4" xcrun -sdk iphoneos metal -fmodules-cache-path="$module_cache" "$metal_file" -o "${stem}_iphoneos.metallib"
    count=$((count + 1))
  done < <(find "$dir" -maxdepth 1 -name '*.metal' -print | LC_ALL=C sort)
  if [[ "$count" -eq 0 ]]; then
    echo "no .metal files found in $dir" >&2
    exit 1
  fi
  echo "compiled $count Metal sources in $dir"
}

package_metallibs() {
  local dir="$1"
  local output="$2"
  local count=0
  printf '\n' > "$output"
  while IFS= read -r metallib_file; do
    python3 "$PACKAGER" "$metallib_file" >> "$output"
    count=$((count + 1))
  done < <(find "$dir" -maxdepth 1 -name '*.metallib' -print | LC_ALL=C sort)
  if [[ "$count" -eq 0 ]]; then
    echo "no .metallib files found in $dir" >&2
    exit 1
  fi
  echo "packaged $count Metal libraries from $dir"
}

render_selector() {
  local kind="$1"
  local generator="$2"
  local output="$3"
  local branches="$WORK_DIR/$kind.branches"
  "$generator" --emit-selector > "$branches"
  python3 - "$kind" "$branches" "$output" <<'PY'
import sys

kind, branches_path, output_path = sys.argv[1:]
with open(branches_path, "r", encoding="utf-8") as f:
    branches = f.read()

branches = branches.lstrip("\n")
if "  } else if" not in branches:
    raise SystemExit(f"selector generator for {kind} did not emit any branches")
branches = branches.replace("  } else if", "  if", 1)

if kind == "attention":
    prefix = """#include "AttentionKernel.hpp"
#include <algorithm>
extern "C" {
#include "AttentionKernel+Precompiled.inc"
#include <simd/simd.h>
}

MTL::Library* AttentionKernel::findPrecompiledLibrary(AttentionKernelDescriptor descriptor, MTL::Device *const device, NS::Error **error) const noexcept {
  if (transposeState[AttentionOperand::Q].value_or(true) ||
      transposeState[AttentionOperand::K].value_or(true) ||
      transposeState[AttentionOperand::V].value_or(true) ||
      transposeState[AttentionOperand::O].value_or(true) ||
      transposeState[AttentionOperand::dO].value_or(true) ||
      transposeState[AttentionOperand::dV].value_or(true) ||
      transposeState[AttentionOperand::dK].value_or(true) ||
      transposeState[AttentionOperand::dQ].value_or(true) ||
      !leadingDimensions[AttentionOperand::Q].value_or(true) ||
      !leadingDimensions[AttentionOperand::K].value_or(true) ||
      !leadingDimensions[AttentionOperand::V].value_or(true) ||
      !leadingDimensions[AttentionOperand::O].value_or(true) ||
      !leadingDimensions[AttentionOperand::dO].value_or(true) ||
      !leadingDimensions[AttentionOperand::dV].value_or(true) ||
      !leadingDimensions[AttentionOperand::dK].value_or(true) ||
      !leadingDimensions[AttentionOperand::dQ].value_or(true)) { // Only precompiled versions with transposeState = false and leadingDimensions = true.
    return 0;
  }
  // Not low precision inputs.
  if (memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::FP32 ||
      memoryPrecisions[AttentionOperand::K].value() == GEMMOperandPrecision::FP32 ||
      memoryPrecisions[AttentionOperand::V].value() == GEMMOperandPrecision::FP32 ||
      memoryPrecisions[AttentionOperand::dO].value() == GEMMOperandPrecision::FP32) {
    return 0;
  }
  bool isBF16 = memoryPrecisions[AttentionOperand::Q].value() == GEMMOperandPrecision::BF16 ||
    memoryPrecisions[AttentionOperand::K].value() == GEMMOperandPrecision::BF16 ||
    memoryPrecisions[AttentionOperand::V].value() == GEMMOperandPrecision::BF16 ||
    memoryPrecisions[AttentionOperand::dO].value() == GEMMOperandPrecision::BF16;
  bool lowPrecisionIntermediates = memoryPrecisions[AttentionOperand::L].value() != GEMMOperandPrecision::FP32 ||
    memoryPrecisions[AttentionOperand::D].value() != GEMMOperandPrecision::FP32;
"""
elif kind == "gemm":
    prefix = """#include "GEMMKernel.hpp"
#include "GEMMHeaders.hpp"
#include <algorithm>
extern "C" {
#include "GEMMKernel+Precompiled.inc"
#include <simd/simd.h>
}

MTL::Library* GEMMKernel::findPrecompiledLibrary(GEMMKernelDescriptor descriptor, MTL::Device *const device, NS::Error **error) const noexcept {
"""
else:
    raise SystemExit(f"unknown selector kind: {kind}")

with open(output_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(prefix)
    f.write(branches)
    f.write("  }\n  return 0;\n}\n")
PY
}

compare_or_update() {
  local generated="$1"
  local target="$2"
  if [[ "$CHECK" -eq 1 ]]; then
    if cmp -s "$generated" "$target"; then
      echo "match: $target"
    else
      echo "differs: $target" >&2
      diff -u "$target" "$generated" | sed -n '1,160p' || true
      return 1
    fi
  else
    mv "$generated" "$target"
    echo "updated: $target"
  fi
}

build_generators

if [[ "$SKIP_COMPILE" -eq 0 ]]; then
  emit_sources
  compile_metals "$ATTENTION_DIR"
  compile_metals "$GEMM_DIR"
fi

render_selector attention "$BUILD_DIR/attention_kernel_gen" "$WORK_DIR/AttentionKernel+Precompiled.cpp"
render_selector gemm "$BUILD_DIR/gemm_kernel_gen" "$WORK_DIR/GEMMKernel+Precompiled.cpp"
package_metallibs "$ATTENTION_DIR" "$WORK_DIR/AttentionKernel+Precompiled.inc"
package_metallibs "$GEMM_DIR" "$WORK_DIR/GEMMKernel+Precompiled.inc"

compare_or_update "$WORK_DIR/AttentionKernel+Precompiled.cpp" "$ATTENTION_SELECTOR"
compare_or_update "$WORK_DIR/GEMMKernel+Precompiled.cpp" "$GEMM_SELECTOR"
compare_or_update "$WORK_DIR/AttentionKernel+Precompiled.inc" "$ATTENTION_INC"
compare_or_update "$WORK_DIR/GEMMKernel+Precompiled.inc" "$GEMM_INC"

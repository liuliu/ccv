#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
target="${1:-gemm_scaffold}"
make "$target"
./"$target"

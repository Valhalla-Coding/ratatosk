#!/usr/bin/env bash
# Copies LM Studio's newest CUDA llama-server and its CUDA libraries into Ratatosk's own folder,
# so Ratatosk keeps working after LM Studio is removed. Run on Idun as the user that owns ~/.lmstudio.
# Needs about 1.2 GB free. Usage: scripts/install_engine.sh [destination]   (default ~/ratatosk/engine)
set -euo pipefail
backends="${LMS_HOME:-$HOME/.lmstudio}/extensions/backends"
dest="${1:-$HOME/ratatosk/engine}"
backend=$(ls -d "$backends"/llama.cpp-linux-x86_64-nvidia-cuda-avx2-* | sort -V | tail -1)
vendor="$backends/vendor/linux-llama-cuda-vendor-v1"
[ -x "$backend/llama-server" ] || { echo "no llama-server in $backend" >&2; exit 1; }
[ -d "$vendor" ] || { echo "no CUDA libraries at $vendor" >&2; exit 1; }
mkdir -p "$dest"
rm -rf "$dest/backend" "$dest/vendor"
cp -r "$backend" "$dest/backend"
cp -r "$vendor" "$dest/vendor"
LD_LIBRARY_PATH="$dest/backend:$dest/vendor" "$dest/backend/llama-server" --version
echo "engine installed in $dest"

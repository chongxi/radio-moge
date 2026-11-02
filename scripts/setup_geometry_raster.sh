#!/usr/bin/env bash
# Isolated rasterization dependency; never modify the model/preview environment.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: setup_geometry_raster.sh CORPUS_ROOT [NEW_PACKAGE_DIRECTORY_NAME]" >&2
    exit 2
fi
corpus_root="$1"
package_name="${2:-raster-packages}"
[[ "$package_name" =~ ^raster-packages(-[0-9]+)?$ ]]
test -f "$corpus_root/plan.json"
test -x "$corpus_root/prep-venv/bin/python"
test ! -e "$corpus_root/$package_name"
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Dedicated cache avoids ownership conflicts with a different user's uv cache.
export UV_CACHE_DIR="/tmp/geometry-raster-uv-$UID"
uv pip install --python "$corpus_root/prep-venv/bin/python" \
    --target "$corpus_root/$package_name" --no-deps --no-build-isolation \
    'nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae'

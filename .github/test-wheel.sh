#!/usr/bin/env bash
# Install the xdmf wheel from a directory and run the Python tests against it.
# Usage, from the repository root: .github/test-wheel.sh <python> <wheel directory>
set -euo pipefail

python="$1"
wheels="$2"

"$python" -m pip install --upgrade pip # RHEL 8's pip 20.2 doesn't know the manylinux_2_28 tag
"$python" -m pip install pytest h5py numpy
"$python" -m pip install --no-index --find-links "$wheels" xdmf
"$python" -c "import xdmf; assert xdmf.is_hdf5_enabled(), 'the wheel was built without HDF5'"
"$python" -m pytest python/tests

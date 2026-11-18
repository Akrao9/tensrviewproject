#!/usr/bin/env bash
# Clean install of `tensorquill-server` from TestPyPI in a Linux container.
#
#   server/packaging/testpypi_install.sh [version]
#
# Run it after the wheels are on TestPyPI. With no version it uses the one in server/Cargo.toml.
# It needs Docker (or set CONTAINER=podman). The container has the machine's own architecture,
# so on an Apple Silicon Mac it tests the arm64 wheel and on an x86_64 machine the x86_64 one;
# set PLATFORM=linux/amd64 or PLATFORM=linux/arm64 to choose (the other one runs under emulation).
#
# It passes only if pip took the manylinux_2_17 wheel, not the fallback, and the program runs.
# Only TestPyPI is used as the index: the package has no dependencies, so nothing can come from
# anywhere else. A second container (Alpine, musl) must get the fallback wheel and no program.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VERSION="${1:-$(sed -n '/^\[package\]/,/^\[/s/^version *= *"\(.*\)"/\1/p' "$HERE/../Cargo.toml")}"
CONTAINER="${CONTAINER:-docker}"
INDEX="https://test.pypi.org/simple/"
PLATFORM_ARGS=()
if [ -n "${PLATFORM:-}" ]; then PLATFORM_ARGS=(--platform "$PLATFORM"); fi

echo "tensorquill-server==$VERSION from $INDEX"

"$CONTAINER" run --rm ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} -e VERSION="$VERSION" -e INDEX="$INDEX" \
  -v "$HERE/check_wheel.py:/check_wheel.py:ro" python:3.12-slim bash -ec '
    python -m venv /tmp/venv
    /tmp/venv/bin/pip install --no-cache-dir --index-url "$INDEX" "tensorquill-server==$VERSION"
    /tmp/venv/bin/python /check_wheel.py installed "manylinux_2_17_$(uname -m)"
    /tmp/venv/bin/tensorquill-server --help
    /tmp/venv/bin/tensorquill-server detect --help >/dev/null
    echo "glibc container: the manylinux_2_17 wheel installed and the program ran"
  '

"$CONTAINER" run --rm ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} -e VERSION="$VERSION" -e INDEX="$INDEX" \
  -v "$HERE/check_wheel.py:/check_wheel.py:ro" python:3.12-alpine sh -ec '
    python -m venv /tmp/venv
    /tmp/venv/bin/pip install --no-cache-dir --index-url "$INDEX" "tensorquill-server==$VERSION"
    /tmp/venv/bin/python /check_wheel.py installed any
    echo "musl container: the fallback wheel installed, with no program"
  '

echo "PASSED: tensorquill-server $VERSION from TestPyPI"

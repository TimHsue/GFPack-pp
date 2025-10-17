#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CXX="${CXX:-g++}"
PY_INCLUDES="$(python3 -m pybind11 --includes)"
PY_SUFFIX="$(python3-config --extension-suffix)"

COMMON_FLAGS=(
  -std=c++17
  -O3
  -march=native
  -finline-functions
  -funroll-loops
  -Wall
  -fPIC
  -fopenmp
)

"${CXX}" "${COMMON_FLAGS[@]}" ${PY_INCLUDES} \
  util.cpp clipper.engine.cpp clipper.offset.cpp clipper.rectclip.cpp \
  -shared -o "calutil${PY_SUFFIX}"

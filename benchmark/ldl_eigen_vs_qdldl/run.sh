#!/usr/bin/env bash
# Build and run the Eigen-vs-QDLDL LDL^T benchmark. Needs third_party/eigen and third_party/qdldl.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TP=../../third_party
mkdir -p build
gcc -O3 -DNDEBUG -Igenerated -I$TP/qdldl/include -c $TP/qdldl/src/qdldl.c -o build/qdldl.o
g++ -std=c++17 -O3 -DNDEBUG -I$TP/eigen -Igenerated -I$TP/qdldl/include bench.cpp build/qdldl.o -o build/bench
./build/bench

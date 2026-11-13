#!/usr/bin/env python3
"""Compare the three polishing methods on the built altro examples."""

import argparse
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass


METHODS = ("riccati", "kkt-eigen", "kkt-qdldl")


@dataclass(frozen=True)
class Sample:
    milliseconds: float
    steps: int
    al_iterations: int
    before: float
    after: float
    dynamics: float


def run_example(executable, output_dir, method):
    result = subprocess.run([executable, output_dir, method, "--summary"], capture_output=True, text=True, check=True)

    def match(pattern):
        found = re.search(pattern, result.stdout, re.MULTILINE)
        if found is None:
            raise ValueError(f"{executable.name}/{method}: missing {pattern}\n{result.stdout}")
        return found

    backend = match(r"^ALTRO projection method : (\S+)$")[1]
    projection = match(r"^ALTRO projection : (\d+) steps, (\S+) ms$")
    sample = Sample(
        milliseconds=float(projection[2]),
        steps=int(projection[1]),
        al_iterations=int(match(r"^AL-iLQR iterations : (\d+),")[1]),
        before=float(match(r"^AL-iLQR max_violation : (\S+)$")[1]),
        after=float(match(r"^ALTRO solver max violation : (\S+)$")[1]),
        dynamics=float(match(r"^ALTRO dynamics residual : (\S+)$")[1]),
    )
    if backend != method or sample.steps == 0:
        raise ValueError(f"{executable.name}/{method}: no accepted steps for the selected method\n{result.stdout}")
    if not all(math.isfinite(x) for x in (sample.milliseconds, sample.before, sample.after, sample.dynamics)):
        raise ValueError(f"{executable.name}/{method}: non-finite result: {sample}")
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path, help="CMake Release build containing alilqr and mp")
    parser.add_argument("--runs", type=int, default=31, help="timed runs per example and method (default: 31)")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    print("example,method,median_polishing_ms,steps,violation_before,violation_after,max_dynamics_residual")
    with tempfile.TemporaryDirectory(prefix="altro-projection-") as output_dir:
        for name in ("alilqr", "mp"):
            executable = args.build_dir.resolve() / name
            samples = {method: [] for method in METHODS}
            for method in METHODS:
                run_example(executable, output_dir, method)
            for i in range(args.runs):
                offset = i % len(METHODS)
                for method in METHODS[offset:] + METHODS[:offset]:
                    samples[method].append(run_example(executable, output_dir, method))

            all_samples = [sample for runs in samples.values() for sample in runs]
            if len({(sample.al_iterations, sample.before) for sample in all_samples}) != 1:
                raise ValueError(f"{name}: AL-iLQR results differed across runs")
            for method, runs in samples.items():
                results = {(sample.steps, sample.after) for sample in runs}
                if len(results) != 1:
                    raise ValueError(f"{name}/{method}: polishing results differed across runs: {results}")
                steps, after = results.pop()
                elapsed = statistics.median(sample.milliseconds for sample in runs)
                dynamics = max(sample.dynamics for sample in runs)
                print(f"{name},{method},{elapsed:.3f},{steps},{runs[0].before:.6g},{after:.6g},{dynamics:.6g}")


if __name__ == "__main__":
    main()

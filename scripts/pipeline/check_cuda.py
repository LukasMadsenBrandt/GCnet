#!/usr/bin/env python3
"""Inspect CUDA compatibility for optional pipeline GPU backends."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.analysis.cuda_environment import collect_cuda_environment_report, write_cuda_environment_report


def main() -> None:
    """Print or write a CUDA readiness report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--output-file", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.output_file:
        output = write_cuda_environment_report(args.output_file)
        print(f"CUDA compatibility report written to {output}")
        return

    report = collect_cuda_environment_report()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print(f"Python: {report.python}")
    print(f"Platform: {report.platform}")
    print(f"nvidia-smi: {'available' if report.nvidia_smi_available else 'missing'}")
    if report.nvidia_smi_summary:
        print(f"nvidia-smi summary: {report.nvidia_smi_summary}")
    for package in report.packages:
        version = f" {package.version}" if package.version else ""
        print(f"{package.name}: {'available' if package.available else 'missing'}{version}")
    print(f"CuPy GPU check: {'ok' if report.cupy_gpu_check_ok else 'not ready'}")
    if report.cupy_error:
        print(f"CuPy error: {report.cupy_error}")
    if report.cugraph_error:
        print(f"cuGraph error: {report.cugraph_error}")
    for device in report.devices:
        print(
            f"GPU {device.index}: {device.name}, "
            f"compute capability {device.compute_capability}, memory {device.memory_gb:.1f} GB"
        )
    print(f"GPU GC ready: {report.cupy_gc_ready}")
    print(f"GPU consensus ready: {report.cugraph_consensus_ready}")
    if report.notes:
        print("Notes:")
        for note in report.notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()

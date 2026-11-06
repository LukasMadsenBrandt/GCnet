#!/usr/bin/env python3
"""Plan, pilot, run or resume exhaustive genome FDR with the shared config."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.fdr import load_samples
from gene_analysis.pipeline.runner import PipelineConfig, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="store_true", help="CPU-only input validation and storage estimate; no fitting or output.")
    parser.add_argument("--pilot-shards", type=int, help="Commit at most this many new fitting shards, then stop; resumable.")
    parser.add_argument("--stage", choices=["fit", "adjust", "report"], help="Run through this stage, including missing prerequisites.")
    parser.add_argument("--backend", choices=["cpu_numpy", "gpu_cuda"], help="Override fitting backend; completed fits are reused.")
    args = parser.parse_args()
    config = PipelineConfig.from_yaml(args.config)
    if config.analysis_mode != "fdr" or config.analysis.workflow not in {"genome_all_pairs", "seed_genome"}:
        parser.error("This entry point requires analysis.mode=fdr and workflow=genome_all_pairs or seed_genome.")
    overrides = {}
    if args.pilot_shards is not None:
        if args.pilot_shards < 1:
            parser.error("--pilot-shards must be positive")
        overrides.update(genome_shards_per_invocation=args.pilot_shards, genome_stage="fit")
    if args.stage:
        if args.pilot_shards is not None and args.stage != "fit":
            parser.error("--pilot-shards only supports --stage fit")
        overrides["genome_stage"] = args.stage
    if args.backend:
        overrides["backend"] = args.backend
    config = replace(config, analysis=replace(config.analysis, **overrides))
    config.validate()
    if args.plan:
        values, times, genes, subset, _ = load_samples(config)
        tests = (len(subset)*(2*len(genes)-len(subset)-1) if config.analysis.workflow == "seed_genome"
                 else len(genes) * (len(genes)-1))
        ancestor = config.run_dir
        while not ancestor.exists():
            ancestor = ancestor.parent
        print(json.dumps(dict(genes=len(genes), samples=len(times), tests=tests,
              expression_array_bytes=values.nbytes, output_directory=str(config.run_dir / "genome_fdr"),
              uncompressed_raw_sorted_adjusted_bytes=96*tests,
              free_disk_bytes=shutil.disk_usage(ancestor).free,
              note="96 bytes/test is a column-payload estimate, not a capacity guarantee; allow sort spill, Parquet overhead and headroom. No GPU used."), indent=2))
        return
    for name, path in run_pipeline(config).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

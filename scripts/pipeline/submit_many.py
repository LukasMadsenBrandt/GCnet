#!/usr/bin/env python3
"""Validate and launch multiple independent pipeline configs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_analysis.pipeline.runner import PipelineConfig


def load_configs(paths: list[str], *, check_existing_results: bool = False) -> list[tuple[Path, PipelineConfig]]:
    """Load pipeline configs and ensure each run name is unique."""
    loaded = [(Path(path), PipelineConfig.from_yaml(path)) for path in paths]
    seen: dict[str, Path] = {}
    duplicates: list[str] = []
    existing_results: list[str] = []
    for path, cfg in loaded:
        if cfg.run_name in seen:
            duplicates.append(f"{cfg.run_name}: {seen[cfg.run_name]} and {path}")
        seen[cfg.run_name] = path
        if check_existing_results and cfg.run_dir.exists():
            existing_results.append(f"{cfg.run_name}: {cfg.run_dir}")
    if duplicates:
        raise ValueError("Duplicate run_name values are not safe for parallel execution:\n" + "\n".join(duplicates))
    if existing_results:
        raise ValueError("Existing result folders found for configured run_name values:\n" + "\n".join(existing_results))
    return loaded


def pipeline_command(config_path: str | Path, *, python: str = "python3") -> list[str]:
    """Return the canonical command for one pipeline config."""
    return [python, "scripts/pipeline/run_pipeline.py", "--config", str(config_path)]


def write_slurm_job_scripts(
    configs: list[tuple[Path, PipelineConfig]],
    output_dir: str | Path,
    *,
    job_name: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    python: str,
    extra_sbatch: list[str] | None = None,
) -> list[Path]:
    """Write one SLURM job script per pipeline config plus a submit-all script."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extra = extra_sbatch or []
    scripts: list[Path] = []
    for config_path, cfg in configs:
        script = output / f"{cfg.run_name}.slurm"
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}_{cfg.run_name}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output=logs/{cfg.run_name}_%j.out",
            f"#SBATCH --error=logs/{cfg.run_name}_%j.err",
        ]
        lines.extend(f"#SBATCH {item}" for item in extra)
        lines.extend(
            [
                "",
                "set -euo pipefail",
                "mkdir -p logs",
                "",
                f"echo {shlex.quote(f'Running {config_path}')}",
                shlex.join(pipeline_command(config_path, python=python)),
                "",
            ]
        )
        script.write_text("\n".join(lines), encoding="utf-8")
        scripts.append(script)

    submit_all = output / "submit_all.sh"
    submit_all.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                *[f"sbatch {shlex.quote(str(script))}" for script in scripts],
                "",
            ]
        ),
        encoding="utf-8",
    )
    scripts.append(submit_all)
    return scripts


def write_slurm_array_script(
    configs: list[tuple[Path, PipelineConfig]],
    output_file: str | Path,
    *,
    job_name: str,
    cpus_per_task: int,
    mem: str,
    time_limit: str,
    python: str,
    extra_sbatch: list[str] | None = None,
) -> Path:
    """Write a SLURM array script where each array task runs one config."""
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_paths = [str(path) for path, _ in configs]
    extra = extra_sbatch or []
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{len(config_paths) - 1}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time_limit}",
        "#SBATCH --output=logs/%x_%A_%a.out",
        "#SBATCH --error=logs/%x_%A_%a.err",
    ]
    lines.extend(f"#SBATCH {item}" for item in extra)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "mkdir -p logs",
            "",
            "CONFIGS=(",
            *[f"  {shlex.quote(path)}" for path in config_paths],
            ")",
            'CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"',
            'echo "Running ${CONFIG}"',
            shlex.join([python, "scripts/pipeline/run_pipeline.py", "--config"]) + ' "$CONFIG"',
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    """Parse CLI arguments and execute the requested multi-config action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="Pipeline YAML configs to run.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for generated commands.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and print commands without running.")
    parser.add_argument("--run-local", action="store_true", help="Run configs sequentially in this shell.")
    parser.add_argument("--write-slurm", default=None, help="Write one SLURM job script per config into this folder.")
    parser.add_argument("--write-slurm-array", default=None, help="Write one SLURM array script to this path.")
    parser.add_argument(
        "--check-existing-results",
        action="store_true",
        help="Fail if any config's results/pipeline/<run_name> folder already exists.",
    )
    parser.add_argument("--job-name", default="gene_pipeline")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument(
        "--sbatch",
        action="append",
        default=[],
        help="Extra SBATCH directive, for example '--partition=gpu'. Can be repeated.",
    )
    args = parser.parse_args()

    if sum(bool(flag) for flag in (args.dry_run, args.run_local, args.write_slurm, args.write_slurm_array)) != 1:
        parser.error("Choose exactly one of --dry-run, --run-local, --write-slurm, or --write-slurm-array.")

    configs = load_configs(args.configs, check_existing_results=args.check_existing_results)
    commands = [pipeline_command(path, python=args.python) for path, _ in configs]

    for (path, cfg), command in zip(configs, commands):
        print(f"{cfg.run_name}: {shlex.join(command)}")

    if args.write_slurm:
        scripts = write_slurm_job_scripts(
            configs,
            args.write_slurm,
            job_name=args.job_name,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_limit=args.time,
            python=args.python,
            extra_sbatch=args.sbatch,
        )
        print(f"SLURM job scripts written to {args.write_slurm}")
        print(f"Submit all with: bash {Path(args.write_slurm) / 'submit_all.sh'}")
        return

    if args.write_slurm_array:
        script = write_slurm_array_script(
            configs,
            args.write_slurm_array,
            job_name=args.job_name,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time_limit=args.time,
            python=args.python,
            extra_sbatch=args.sbatch,
        )
        print(f"SLURM array script written to {script}")
        return

    if args.run_local:
        for command in commands:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

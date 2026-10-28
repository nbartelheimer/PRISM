#!/usr/bin/env python3
"""Run ten independent OSU experiments per IB protocol and metric."""

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

EXPERIMENTS = 10
MODES = {
    "PRISM_eager": "eager",
    "PRISM_rendezvous": "rendezvous",
    "PRISM_no_sync": "no_sync",
    "vanilla": None,
}
COLUMNS = ["message_size", *MODES, "experiment_id"]
# These are set explicitly per rank, after the launcher starts the process.
MODE_VARIABLES = ("PRISM_IB_EAGER_LIMIT",)


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", required=True,
                        help="quoted MPI/Slurm launch command, including exactly two ranks")
    parser.add_argument("--osu-dir", type=Path, default=Path("."),
                        help="directory containing vanilla OSU executables (default: current directory)")
    parser.add_argument("--prism-osu-dir", type=Path,
                        help="directory containing PRISM OSU executables (default: --osu-dir)")
    for name in ("latency", "bandwidth", "prism-latency", "prism-bandwidth",
                 "no-sync-latency", "no-sync-bandwidth"):
        parser.add_argument("--" + name + "-bin", type=Path,
                            help="explicit executable path, overriding directory-based discovery")
    parser.add_argument("--prism-lib-dir", type=Path,
                        help="prepend this directory to LD_LIBRARY_PATH on PRISM ranks")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="set an environment variable on every benchmark rank (repeatable)")
    parser.add_argument("--min-size", type=positive, default=1, help="minimum bytes (default: 1)")
    parser.add_argument("--max-size", type=positive, default=4194304,
                        help="maximum bytes (default: 4194304)")
    parser.add_argument("--iterations", type=positive, help="OSU -i; omit to use OSU defaults")
    parser.add_argument("--warmup", type=positive, help="OSU -x; omit to use OSU defaults")
    parser.add_argument("--window-size", type=positive, default=64,
                        help="bandwidth OSU -W, identical for all variants (default: 64)")
    parser.add_argument("--timeout", type=positive, default=1800,
                        help="seconds allowed per launch (default: 1800)")
    parser.add_argument("--output-dir", type=Path,
                        help="new output directory; defaults to ib-osu-<UTC timestamp>")
    parser.add_argument("--dry-run", action="store_true",
                        help="print all 80 commands without launching jobs or writing results")
    args = parser.parse_args(argv)
    if args.max_size < args.min_size:
        parser.error("--max-size must be at least --min-size")
    if args.window_size > 512:
        parser.error("--window-size exceeds PRISM's 512-request store")
    try:
        args.launcher = shlex.split(args.launcher)
    except ValueError as error:
        parser.error(str(error))
    if not args.launcher:
        parser.error("--launcher cannot be empty")
    args.rank_env = {"FI_PROVIDER": "verbs;ofi_rxm"}
    for assignment in args.env:
        key, sep, value = assignment.partition("=")
        if not sep or not key or not key.replace("_", "a").isascii() or not key.isidentifier():
            parser.error("--env requires a valid NAME=VALUE assignment")
        if key in MODE_VARIABLES or key == "LC_ALL":
            parser.error(f"{key} is controlled by this runner")
        args.rank_env[key] = value
    args.rank_env["LC_ALL"] = "C"
    return args


def executable(directory, override, names):
    if override is not None:
        candidates = [override.expanduser()]
    else:
        candidates = [directory.expanduser() / name for name in names]
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            # Keep a symlink's executable name; resolve only its parent directory.
            return str(path.absolute())
    raise ValueError("Executable not found or not executable: " + ", ".join(map(str, candidates)))


def binaries(args):
    standard = args.osu_dir
    persistent = args.prism_osu_dir or standard
    return {
        "latency": {
            "vanilla": executable(standard, args.latency_bin, ["osu_latency"]),
            "prism": executable(persistent, args.prism_latency_bin, ["osu_latency_persistent"]),
            "no_sync": executable(persistent, args.no_sync_latency_bin,
                                  ["osu_latency_persistent_no_sync"]),
        },
        "bandwidth": {
            "vanilla": executable(standard, args.bandwidth_bin, ["osu_bandwidth", "osu_bw"]),
            "prism": executable(persistent, args.prism_bandwidth_bin, ["osu_bw_persistent"]),
            "no_sync": executable(persistent, args.no_sync_bandwidth_bin,
                                  ["osu_bw_persistent_no_sync"]),
        },
    }


def eager_limit(args, variant):
    if variant == "vanilla":
        return None
    return args.max_size if variant == "PRISM_eager" else 0


def command(args, executables, metric, variant):
    kind = "vanilla" if variant == "vanilla" else "no_sync" if variant == "PRISM_no_sync" else "prism"
    environment = dict(args.rank_env)
    if MODES[variant] is not None:
        # Cover the full requested range, including sizes above the default
        # 4096-byte cutoff. The backend still excludes provider-injectable sizes.
        environment["PRISM_IB_EAGER_LIMIT"] = str(eager_limit(args, variant))
        if args.prism_lib_dir:
            previous = environment.get("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", ""))
            directory = str(args.prism_lib_dir.expanduser().absolute())
            environment["LD_LIBRARY_PATH"] = directory + (":" + previous if previous else "")
    # Run env *inside* the launcher: mode selection reaches both remote ranks
    # even when mpirun/srun does not export the driver's environment by default.
    result = list(args.launcher) + ["env"]
    for name in MODE_VARIABLES:
        result += ["-u", name]
    result += [f"{key}={value}" for key, value in environment.items()]
    result += [executables[metric][kind], "-m", f"{args.min_size}:{args.max_size}"]
    if args.iterations is not None:
        result += ["-i", str(args.iterations)]
    if args.warmup is not None:
        result += ["-x", str(args.warmup)]
    if metric == "bandwidth":
        result += ["-W", str(args.window_size)]
    return result


def parse_measurements(path, expected_sizes):
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if not fields[0].lstrip("-").isdigit():
            continue  # OSU banners and informational lines.
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected 'size value', got {line!r}")
        size, raw_value = int(fields[0]), fields[1]
        try:
            value = Decimal(raw_value)
        except InvalidOperation:
            raise ValueError(f"{path}:{number}: invalid measurement {raw_value!r}") from None
        if size < 0 or not value.is_finite() or value < 0 or size in values:
            raise ValueError(f"{path}:{number}: invalid or duplicate measurement: {line!r}")
        values[size] = raw_value  # Preserve OSU's exact printed precision.
    if set(values) != expected_sizes:
        missing = sorted(expected_sizes - set(values))
        extra = sorted(set(values) - expected_sizes)
        raise ValueError(f"{path}: unexpected size grid; missing={missing}, extra={extra}. "
                         "Use normal two-column OSU output without rank prefixes.")
    return values


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_manifest(output, manifest):
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")


def run_command(argv, stdout_path, stderr_path, timeout):
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            # Terminate the launcher and local children on timeout/Ctrl-C.
            # The site's launcher is responsible for tearing down remote ranks.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise


def main(argv=None):
    args = arguments(argv)
    try:
        executables = binaries(args)
        if not shutil.which(args.launcher[0]):
            raise ValueError(f"Launcher not found: {args.launcher[0]}")
        if args.prism_lib_dir and not (args.prism_lib_dir.expanduser() / "libprism.so").is_file():
            raise ValueError("--prism-lib-dir must contain the IB build's libprism.so")
        size, expected = args.min_size, set()
        while size <= args.max_size:
            expected.add(size)
            size *= 2
        jobs = []
        variants = list(MODES)
        for experiment in range(1, EXPERIMENTS + 1):
            shift = (experiment - 1) % len(variants)
            order = variants[shift:] + variants[:shift]
            for metric in executables:
                for variant in order:
                    jobs.append((experiment, metric, variant,
                                 command(args, executables, metric, variant)))
        if args.dry_run:
            for experiment, metric, variant, argv in jobs:
                print(f"# experiment_id={experiment} metric={metric} variant={variant}")
                print(shlex.join(argv))
            return 0
        output = (args.output_dir or Path("ib-osu-" + datetime.now(timezone.utc).strftime(
            "%Y%m%d-%H%M%S-%f"))).expanduser().absolute()
        output.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    (output / "logs").mkdir()
    for metric in executables:
        with (output / f"{metric}.csv").open("w", newline="") as stream:
            csv.writer(stream).writerow(COLUMNS)
    manifest = {
        "status": "running", "started": timestamp(), "experiments": EXPERIMENTS,
        "columns": COLUMNS, "protocol_modes": MODES, "binaries": executables,
        "PRISM_IB_EAGER_LIMIT": {name: eager_limit(args, name) for name in MODES},
        "eager_selection": "provider inject limit < message bytes <= PRISM_IB_EAGER_LIMIT",
        "units": {"message_size": "bytes", "latency": "microseconds", "bandwidth": "MB/s"},
        "message_sizes": sorted(expected), "timeout_seconds": args.timeout,
        "aggregation": "none across launches; retain each OSU-reported value verbatim",
        "runs": [],
    }
    save_manifest(output, manifest)
    print(f"Results: {output}", flush=True)
    results = {}
    try:
        for experiment, metric, variant, argv in jobs:
            stem = f"{metric}-{experiment:02d}-{variant}"
            stdout = output / "logs" / f"{stem}.stdout"
            stderr = output / "logs" / f"{stem}.stderr"
            record = {
                "experiment_id": experiment, "metric": metric, "variant": variant,
                "command": argv, "started": timestamp(),
                "stdout": str(stdout.relative_to(output)), "stderr": str(stderr.relative_to(output)),
                "status": "running",
            }
            manifest["runs"].append(record)
            save_manifest(output, manifest)
            print(f"[{len(manifest['runs'])}/{len(jobs)}] experiment {experiment}: {metric}/{variant}",
                  flush=True)
            record["returncode"] = run_command(argv, stdout, stderr, args.timeout)
            record["finished"] = timestamp()
            if record["returncode"]:
                raise RuntimeError(f"Benchmark exited with {record['returncode']}; see {stdout} and {stderr}")
            values = parse_measurements(stdout, expected)
            block = results.setdefault((experiment, metric), {})
            block[variant] = values
            record["status"] = "complete"
            if len(block) == len(MODES):
                # A complete four-variant block only; never fabricate missing cells.
                with (output / f"{metric}.csv").open("a", newline="") as stream:
                    writer = csv.writer(stream)
                    for size in sorted(expected):
                        writer.writerow([size, *(block[name][size] for name in MODES), experiment])
                del results[(experiment, metric)]
            save_manifest(output, manifest)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error) or "Interrupted"
        manifest["finished"] = timestamp()
        if manifest["runs"]:
            manifest["runs"][-1]["status"] = "failed"
            manifest["runs"][-1]["finished"] = timestamp()
        save_manifest(output, manifest)
        print(f"Error: {manifest['error']}\nPartial results and logs: {output}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    manifest["status"] = "complete"
    manifest["finished"] = timestamp()
    save_manifest(output, manifest)
    print(f"Completed {len(jobs)} launches.\n{output / 'latency.csv'}\n{output / 'bandwidth.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compile, validate, and execute a resumable density/size study.

Independent subprocesses only parallelize numerical repetitions. Seeds depend
on parameter values and trial number, never job order or number of workers.
"""
import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NS = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
RHOS = sorted(set(range(10, 41, 2)) | {35})
KS = [1, 5, 10, 25]


def repetitions(n):
    return 50 if n <= 8000 else (30 if n == 16000 else 20)


def valid_shard(path, n, rho, start, count):
    if not path.exists():
        return False
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(trial, k) for trial in range(start, start + count) for k in [0] + KS}
    actual = {(int(r["trial"]), int(r["k"])) for r in rows}
    return (len(rows) == len(expected) and actual == expected
            and all(int(r["n"]) == n and float(r["rho"]) == rho
                    and int(r["truncated"]) == 0 for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "scaling")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = args.output_dir / "shards"
    shards.mkdir(exist_ok=True)
    binary = args.output_dir / "hks_density_scaling"
    compile_command = ["g++", "-O3", "-std=c++17", "-Wall", "-Wextra", "-pedantic",
                       str(ROOT / "src" / "hks_extended_experiments.cpp"), "-o", str(binary)]
    subprocess.run(compile_command, check=True)
    subprocess.run([str(binary), "--design", "selftest"], check=True)
    jobs = [(n, rho, start, min(5, repetitions(n) - start + 1))
            for n in NS for rho in RHOS for start in range(1, repetitions(n) + 1, 5)]
    # Start larger jobs first to avoid a long sequential tail.
    jobs.sort(key=lambda job: (-job[0], -job[1], job[2]))
    manifest = {
        "design": "density_scaling", "n_values": NS, "rho_values": RHOS,
        "k_values": KS, "median_control": True,
        "repetitions_by_n": {str(n): repetitions(n) for n in NS},
        "profile_generator": "std::mt19937_64; GCC libstdc++ uniform_real_distribution",
        "activation_generator": "SplitMix64 with rejection sampling",
        "master_seed_formula": "splitmix_hash(0x20261007 ^ fnv1a('density_scaling'))",
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
        "compile_command": compile_command,
        "bootstrap_seed": 20261008, "bootstrap_repetitions": 2000,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def execute(job):
        n, rho, start, count = job
        path = shards / f"n{n}_rho{rho}_trial{start:03d}.csv"
        if valid_shard(path, n, rho, start, count):
            return job, True
        command = [str(binary), "--design", "density_scaling", "--ns", str(n),
                   "--rhos", str(rho), "--ks", ",".join(map(str, KS)),
                   "--trial-start", str(start), "--reps", str(count), "--output", str(path)]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0 or not valid_shard(path, n, rho, start, count):
            raise RuntimeError(f"failed job {job}: {result.stderr}")
        return job, False

    start_time = time.monotonic()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            job, reused = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                print(f"completed {completed}/{len(jobs)} shards; elapsed {time.monotonic()-start_time:.0f}s",
                      flush=True)

    all_rows = []
    for n, rho, start, count in sorted(jobs):
        path = shards / f"n{n}_rho{rho}_trial{start:03d}.csv"
        with path.open(newline="") as handle:
            all_rows.extend(csv.DictReader(handle))
    all_rows.sort(key=lambda row: (int(row["n"]), float(row["rho"]), int(row["trial"]), int(row["k"])))
    raw_path = args.output_dir / "hks_density_scaling_raw.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    manifest["raw_runs"] = len(all_rows)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(all_rows)} runs to {raw_path}", flush=True)


if __name__ == "__main__":
    main()

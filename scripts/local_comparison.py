"""Run prespecified local comparison seeds, retaining every run and failure."""

import argparse
import copy
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from research.fedgcdr_akf.data import load_prepared, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path, help="Output filename prefix for aggregate artifacts")
    args = parser.parse_args()
    config = load_prepared(args.config)
    seeds = config["comparison"]["seeds"]
    workers = config["comparison"]["workers"]
    if not seeds or len(set(seeds)) != len(seeds) or workers < 1:
        raise ValueError("Use distinct seeds and a positive worker count.")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "comparison_config.json", config)
    jobs = []
    for seed in seeds:
        seeded = copy.deepcopy(config)
        seeded["training"]["seed"] = seed
        path = args.output / f"config_seed_{seed}.json"
        write_json(path, seeded)
        jobs.append((seed, path))
    temp_dir = ROOT / ".cache/tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "TMPDIR": str(temp_dir), "SQLITE_TMPDIR": str(temp_dir)})
    completed, failed = [], []

    def save_status(status):
        write_json(args.output / "status.json", {"status": status, "completed": completed,
                                                "failed": failed, "seeds": seeds})

    save_status("running")

    def execute(job):
        seed, path = job
        console = args.output / f"seed_{seed}.console.log"
        command = [sys.executable, "-m", "research.fedgcdr_akf", "run", "--config", str(path.resolve()),
                   "--prepared", str(args.prepared.resolve()), "--output", str((args.output / f"seed_{seed}").resolve())]
        print(json.dumps({"seed": seed, "status": "started", "console": str(console)}), flush=True)
        with console.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        return seed, result.returncode

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(execute, job) for job in jobs]
        for future in as_completed(futures):
            seed, returncode = future.result()
            (completed if returncode == 0 else failed).append(seed)
            save_status("running")
            print(json.dumps({"seed": seed, "returncode": returncode, "completed": completed, "failed": failed}), flush=True)
    if failed:
        save_status("failed")
        raise SystemExit(f"Failed seeds retained for inspection: {failed}; see console logs.")
    save_status("aggregating")
    from research.fedgcdr_akf.comparison import summarize
    try:
        summarize(args.output, args.prepared, args.report)
    except Exception:
        save_status("report_failed")
        raise
    save_status("complete")
    print(json.dumps({"status": "complete", "report": str(args.report.with_suffix('.md'))}), flush=True)


if __name__ == "__main__":
    main()

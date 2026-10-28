import argparse
import json
from pathlib import Path

from .data import prepare


def main():
    parser = argparse.ArgumentParser(description="FedGCDR module adaptation and adaptive knowledge fusion")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "run":
            command.add_argument("--prepared", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "prepare":
        summary = prepare(config["data"], args.output)
        print(json.dumps({"prepared": str(args.output), "users": summary["users"],
                          "source_overlap_users": summary["source_overlap_users"]}, ensure_ascii=False))
    else:
        from .run import run
        summary = run(config, args.prepared, args.output)
        print(json.dumps({"completed": str(args.output), "modes": list(summary["outcomes"]),
                          "elapsed_seconds": summary["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

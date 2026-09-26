"""Command-line validation tools that do not require a CARLA server."""

import argparse
import json

from .dataset import validate_dataset
from .scenario import load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cornersim")
    commands = parser.add_subparsers(dest="command", required=True)
    scenario = commands.add_parser("validate-scenario")
    scenario.add_argument("path")
    dataset = commands.add_parser("validate-dataset")
    dataset.add_argument("path")
    args = parser.parse_args(argv)
    if args.command == "validate-scenario":
        parsed = load_scenario(args.path)
        print(json.dumps({"valid": True, "actions": len(parsed.actions), "seed": parsed.seed}))
        return 0
    report = validate_dataset(args.path)
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

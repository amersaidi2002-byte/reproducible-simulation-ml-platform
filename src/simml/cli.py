"""Local commands for data, training, evaluation, export and run inspection."""

import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    data = sub.add_parser("dataset")
    data.add_argument("--root", default="work/datasets")
    data.add_argument("--cache", default="work/jobs.sqlite")
    data.add_argument("--families", type=int, default=60)
    data.add_argument("--members", type=int, default=20)
    data.add_argument("--seed", type=int, default=2026)
    data.add_argument("--workers", type=int, default=4)
    for command in ("train", "evaluate"):
        c = sub.add_parser(command)
        c.add_argument("--dataset", required=True)
        c.add_argument("--runs", default="work/runs")
        if command == "train":
            c.add_argument("--epochs", type=int, default=200)
        else:
            c.add_argument("--checkpoint", required=True)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", default="artifacts/bundle")
    runs = sub.add_parser("runs")
    runs.add_argument("--root", default="work/runs")
    args = p.parse_args()
    if args.command == "dataset":
        from .datasets import build_dataset

        path, report = build_dataset(
            args.root, args.cache, args.families, args.members, args.seed, args.workers
        )
        print(json.dumps({"path": str(path), **report}, indent=2))
    elif args.command == "train":
        from .training import train

        print(train(args.dataset, args.runs, {"epochs": args.epochs}))
    elif args.command == "evaluate":
        from .training import evaluate

        print(json.dumps(evaluate(args.checkpoint, args.dataset, args.runs), indent=2))
    elif args.command == "export":
        from .export import export_onnx

        print(json.dumps(export_onnx(args.checkpoint, args.output), indent=2))
    else:
        from .tracking import list_runs

        print(json.dumps(list_runs(args.root), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from model_registry import (
    get_project_root,
    list_runs,
    load_registry,
    promote_alias,
    resolve_alias,
    resolve_repo_path,
    save_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage local BioASQ fine-tuning model metadata.")
    parser.add_argument(
        "--registry-path",
        default="models/registry.json",
        help="Repo-relative path to the model registry JSON file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create the registry file if it does not already exist.")

    list_parser = subparsers.add_parser("list-runs", help="List registered training runs.")
    list_parser.add_argument("--task", default=None, help="Optional task filter, e.g. answer_generation.")
    list_parser.add_argument("--status", default=None, help="Optional status filter, e.g. completed.")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of runs to show.")

    show_parser = subparsers.add_parser("show-run", help="Print the full manifest for one run.")
    show_parser.add_argument("run_id", help="Run identifier from the registry.")

    alias_list_parser = subparsers.add_parser("list-aliases", help="List promoted aliases.")
    alias_list_parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a table.")

    promote_parser = subparsers.add_parser("promote", help="Promote a completed run to a stable alias.")
    promote_parser.add_argument("run_id", help="Run identifier from the registry.")
    promote_parser.add_argument("alias", help="Alias name to point at this run.")

    resolve_parser = subparsers.add_parser("resolve", help="Resolve an alias to its adapter directory.")
    resolve_parser.add_argument("alias", help="Alias name to resolve.")

    return parser.parse_args()


def print_run_summary(run: dict) -> None:
    print(
        f"{run.get('run_id')} | {run.get('status')} | "
        f"{run.get('base_model')} | train={run.get('train_examples')} | "
        f"eval={run.get('eval_examples')} | adapter={run.get('paths', {}).get('adapter_dir')}"
    )


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    registry_path = resolve_repo_path(args.registry_path, project_root=project_root)
    assert registry_path is not None

    if args.command == "init":
        if registry_path.exists():
            print(f"Registry already exists at {registry_path}")
            return
        save_registry(registry_path, load_registry(registry_path))
        print(f"Created registry at {registry_path}")
        return

    if args.command == "list-runs":
        runs = list_runs(registry_path, task=args.task, status=args.status)[: args.limit]
        if not runs:
            print("No runs found.")
            return
        for run in runs:
            print_run_summary(run)
        return

    if args.command == "show-run":
        runs = load_registry(registry_path).get("runs", {})
        run = runs.get(args.run_id)
        if run is None:
            raise KeyError(f"Unknown run_id: {args.run_id}")
        print(json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.command == "list-aliases":
        aliases = load_registry(registry_path).get("aliases", {})
        if args.json:
            print(json.dumps(aliases, ensure_ascii=False, indent=2, sort_keys=True))
            return
        if not aliases:
            print("No aliases found.")
            return
        for alias_name, alias_data in sorted(aliases.items()):
            print(
                f"{alias_name} -> {alias_data.get('run_id')} | "
                f"{alias_data.get('adapter_dir')} | {alias_data.get('base_model')}"
            )
        return

    if args.command == "promote":
        alias_data = promote_alias(registry_path, alias=args.alias, run_id=args.run_id)
        print(
            f"Promoted {alias_data['run_id']} to alias {alias_data['alias']} "
            f"({alias_data.get('adapter_dir')})"
        )
        return

    if args.command == "resolve":
        alias_data = resolve_alias(registry_path, args.alias)
        print(alias_data.get("adapter_dir") or "")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

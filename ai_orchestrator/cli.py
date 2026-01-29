# ai_orchestrator/cli.py
import argparse
import inspect
from pathlib import Path

from .config import load_repo_config
from .workflow import WorkflowRunner

from dotenv import load_dotenv
load_dotenv()

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ai-dev",
        description="AI-driven orchestrator for automated code evolution.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a phase on the current repository.")
    run_p.add_argument(
        "phase",
        type=str,
        help="Phase name, e.g. 'oop_refactor'.",
    )
    run_p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to repository root (default: current directory).",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate outputs and logs only (no branch, no apply, no validation, no commits).",
    )
    run_p.add_argument(
        "--use-plan",
        action="store_true",
        help="Use planner to produce a DAG plan and execute it (requires workflow support).",
    )

    args = parser.parse_args()

    if args.command == "run":
        repo_root = Path(args.repo_root).resolve()
        cfg = load_repo_config(repo_root)
        runner = WorkflowRunner(cfg)

        sig = inspect.signature(runner.run_phase)
        params = sig.parameters

        if args.use_plan and "use_plan" not in params:
            raise SystemExit(
                "This WorkflowRunner.run_phase(...) does not support use_plan yet. "
                "Update workflow.py to accept use_plan, or run without --use-plan."
            )

        kwargs = {"dry_run": bool(args.dry_run)}
        if "use_plan" in params:
            kwargs["use_plan"] = bool(args.use_plan)

        runner.run_phase(args.phase, **kwargs)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

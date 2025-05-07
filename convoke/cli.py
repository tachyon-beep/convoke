import argparse
import os
import logging
import yaml
from convoke.utils import setup_logging, ensure_api_keys
from convoke.store import FileSystemArtifactStore
from convoke.tools import ScopedGetArtifactTool, ScopedSaveArtifactTool
from convoke.workflow import orchestrate_full_workflow
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Convoke CLI entry point")
    parser.add_argument(
        "--project-path",
        type=str,
        required=True,
        help="Path to the project directory (contains config.yaml, output/ folder).",
    )
    parser.add_argument(
        "--requirements",
        type=str,
        required=False,
        help="High-level requirements for the system. (Ignored if project config specifies requirements)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=None, help="Maximum recursion depth"
    )
    parser.add_argument(
        "--max-items", type=int, default=None, help="Maximum items per level"
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Maximum total tasks allowed"
    )
    parser.add_argument(
        "--review-cycles",
        type=int,
        default=None,
        help="Review/refinement cycles per task",
    )
    parser.add_argument(
        "--parse-retries", type=int, default=None, help="Parse retries per task"
    )
    parser.add_argument(
        "--verbose", type=int, default=0, help="Verbosity level (0=INFO, 1=DEBUG)"
    )
    args = parser.parse_args()

    setup_logging()
    ensure_api_keys()
    logger = logging.getLogger("convoke.cli")
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)

    project_path = args.project_path
    config_path = os.path.join(project_path, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Use CLI args if provided, else config, else sensible defaults
    requirements = args.requirements or config.get("requirements")
    max_depth = (
        args.max_depth if args.max_depth is not None else config.get("max_depth", 4)
    )
    max_items = (
        args.max_items if args.max_items is not None else config.get("max_items", 5)
    )
    max_tasks = (
        args.max_tasks if args.max_tasks is not None else config.get("max_tasks", 100)
    )
    review_cycles = (
        args.review_cycles
        if args.review_cycles is not None
        else config.get("review_cycles", 1)
    )
    parse_retries = (
        args.parse_retries
        if args.parse_retries is not None
        else config.get("parse_retries", 1)
    )

    output_dir = os.path.join(project_path, "output")
    artifact_store = FileSystemArtifactStore(output_dir, logger)
    system_read_tool = ScopedGetArtifactTool(
        store=artifact_store, agent_role="System", allowed_read_prefixes=[""]
    )
    system_write_tool = ScopedSaveArtifactTool(
        store=artifact_store, agent_role="System", allowed_write_prefixes=[""]
    )

    logger.info("Starting Convoke workflow...")
    results = orchestrate_full_workflow(
        requirements=requirements,
        max_depth=max_depth,
        max_items=max_items,
        max_tasks=max_tasks,
        verbose=args.verbose,
        logger=logger,
        review_cycles=review_cycles,
        parse_retries=parse_retries,
        output_dir=output_dir,
        artifact_store=artifact_store,
        get_tool=system_read_tool,
        save_tool=system_write_tool,
    )
    if results.get("error"):
        logger.error(f"Workflow terminated due to error: {results['error_msg']}")
        exit(1)
    print("\n===== JSON SUMMARY =====\n")
    import json

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

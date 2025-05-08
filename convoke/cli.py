import argparse
import os
import logging
import yaml
from convoke.utils import setup_logging, ensure_api_keys, ensure_dir_exists
from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Convoke CLI entry point")
    parser.add_argument(
        "--project-path",
        type=str,
        required=True,
        help="Path to the project directory (contains config.yaml, artifacts/, outputs/ and logs/ folders).",
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

    from convoke.agents import (
        create_architect_agent,
        create_architect_reviewer_agent,
        create_module_manager_agent,
        create_module_reviewer_agent,
        create_class_manager_agent,
        create_class_reviewer_agent,
        create_function_manager_agent,
        create_function_reviewer_agent,
    )

    AGENT_TYPE_MAP = {
        "architect": create_architect_agent,
        "architect_reviewer": create_architect_reviewer_agent,
        "module_manager": create_module_manager_agent,
        "module_reviewer": create_module_reviewer_agent,
        "class_manager": create_class_manager_agent,
        "class_reviewer": create_class_reviewer_agent,
        "function_manager": create_function_manager_agent,
        "function_reviewer": create_function_reviewer_agent,
    }

    def hydrate_agents(config):
        for level in config.get("levels", []):
            for task in level.get("tasks", []):
                agent_type = task.pop("agent_type", None)
                if agent_type:
                    if agent_type not in AGENT_TYPE_MAP:
                        raise ValueError(f"Unknown agent_type: {agent_type}")
                    task["agent"] = AGENT_TYPE_MAP[agent_type]()
                if "review" in task and "agent_type" in task["review"]:
                    review_type = task["review"].pop("agent_type")
                    if review_type not in AGENT_TYPE_MAP:
                        raise ValueError(f"Unknown review agent_type: {review_type}")
                    task["review"]["agent"] = AGENT_TYPE_MAP[review_type]()
        return config

    # Hydrate agents in config
    config = hydrate_agents(config)

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

    # Create project directory structure
    logs_dir = os.path.join(project_path, "logs")
    artifacts_dir = os.path.join(project_path, "artifacts")
    outputs_dir = os.path.join(project_path, "outputs")

    # Ensure all required directories exist
    ensure_dir_exists(logs_dir)
    ensure_dir_exists(artifacts_dir)
    ensure_dir_exists(outputs_dir)

    # Set environment variable for artifact path before importing tools
    os.environ["CONVOKE_ARTIFACTS_DIR"] = artifacts_dir

    # Import these after setting the environment variable
    from convoke.store import FileSystemArtifactStore
    from convoke.tools import scoped_get_artifact, scoped_save_artifact
    from convoke.workflow import orchestrate_full_workflow

    # Initialize the artifact store with the project-specific artifacts directory
    artifact_store = FileSystemArtifactStore(artifacts_dir, logger)
    system_read_tool = scoped_get_artifact
    system_write_tool = scoped_save_artifact

    logger.info("Starting Convoke workflow...")

    # Prepare config for the orchestrator
    workflow_config = {
        "max_depth": max_depth,
        "max_items": max_items,
        "max_tasks": max_tasks,
        "review_cycles": review_cycles,
        "parse_retries": parse_retries,
        "store": {
            "base_path": artifacts_dir,
            "logger": logger,
            "agent_role": "orchestrator",
        },
        "artifact_store": artifact_store,  # Include in config
        "get_tool": system_read_tool,  # Include in config
        "save_tool": system_write_tool,  # Include in config
    }

    # Convert output_dir to Path object
    from pathlib import Path

    output_path = Path(outputs_dir)

    # Use hydrated config as workflow_def
    workflow_def = config

    results = orchestrate_full_workflow(
        workflow_def=workflow_def,
        config=workflow_config,
        output_dir=output_path,
        verbose=args.verbose,
        logger=logger,
    )
    if results.get("error"):
        logger.error(f"Workflow terminated due to error: {results['error_msg']}")
        exit(1)
    print("\n===== JSON SUMMARY =====\n")
    import json

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

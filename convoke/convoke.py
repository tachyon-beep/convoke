"""Hierarchical CrewAI programming team with peer reviewers at each level.

This script models a software engineering team with the following roles:
- Systems Architect & Reviewer: Designs the system and is critiqued by a peer.
- Module Manager(s) & Reviewer(s): Each module is designed and reviewed.
- Class Manager(s) & Reviewer(s): Each class is designed and reviewed.
- Function Manager(s) & Reviewer(s): Each function is implemented and reviewed.

Each level dynamically spawns sub-managers and reviewers as needed, enabling recursive, hierarchical codebase design and peer review.

Usage:
    $ python basic.py

Requirements:
    - crewai
    - crewai_tools

Environment Variables:
    Set API keys as needed for external tools (e.g., OPENAI_API_KEY).
"""

from crewai import Agent, Task, Crew, Process
import re
import logging
import argparse
import os
import signal
import json
from typing import List, Tuple, Dict, Any, Optional, Callable
from dotenv import load_dotenv
from pydantic import BaseModel, Field, RootModel
import yaml
from convoke.store import FileSystemArtifactStore
from convoke.tools import ScopedGetArtifactTool, ScopedSaveArtifactTool
from convoke.utils import (
    setup_logging,
    ensure_api_keys,
    ensure_dir_exists,
    write_json_file,
    write_text_file,
    lint_python_code,
)
from convoke.agents import (
    create_architect_agent,
    create_architect_reviewer_agent,
    create_module_manager_agent,
    create_module_reviewer_agent,
    create_class_manager_agent,
    create_class_reviewer_agent,
    create_function_manager_agent,
    create_function_reviewer_agent,
    create_test_developer_agent,
    create_test_reviewer_agent,
    create_architect_task,
    create_architect_review_task,
    create_module_manager_task,
    create_module_review_task,
    create_class_manager_task,
    create_class_review_task,
    create_function_manager_task,
    create_function_review_task,
    create_test_developer_task,
    create_test_review_task,
    ItemDetail,
    ItemListOutput,
)
from convoke.workflow import orchestrate_full_workflow

load_dotenv()


class ItemDetail(BaseModel):
    name: str = Field(description="Name of the item (e.g., module, class, function)")
    description: str = Field(description="Purpose or description of the item")


class ItemListOutput(BaseModel):
    items: List[ItemDetail]


# --- Agent Definitions ---


# --- Task Definitions ---


def parse_numbered_list(text: str, max_items: int = 10) -> List[Tuple[str, str]]:
    """Parse a numbered or bulleted list of items with multi-line descriptions.

    Args:
        text (str): The text to parse.
        max_items (int): Maximum items to parse.
    Returns:
        List[Tuple[str, str]]: List of (name, description) pairs.
    """
    if not isinstance(text, str):
        return []
    try:
        lines = text.splitlines()
    except Exception:
        return []
    items = []
    current_name = None
    current_desc_lines = []
    for line in lines:
        m = re.match(r"^\s*(\d+|[-*])\.?(\s+)([^:]+):?\s*(.*)$", line)
        if m:
            if current_name is not None:
                items.append((current_name, "\n".join(current_desc_lines).strip()))
            current_name = m.group(3).strip()
            desc = m.group(4).strip()
            current_desc_lines = [desc] if desc else []
        elif current_name is not None and not re.match(r"^\s*(\d+|[-*])\.?(\s+)", line):
            # Any non-list-item-start line is a continuation
            current_desc_lines.append(line)
    if current_name is not None:
        items.append((current_name, "\n".join(current_desc_lines).strip()))
    return items[:max_items]


def parse_json_list(
    text: str, max_items: int = 10, logger: Optional[logging.Logger] = None
) -> List[Tuple[str, str]]:
    """Parse a JSON string representing a list of objects into (name, description) tuples."""
    local_logger = logger or logging.getLogger(__name__)
    if not isinstance(text, str) or not text.strip():
        if text is None:
            local_logger.warning("parse_json_list received None input.")
        elif not text.strip():
            local_logger.warning(
                "parse_json_list received empty or whitespace-only string."
            )
        return []
    try:
        match = re.search(r"(\[[\s\S]*\])", text)
        if not match:
            local_logger.warning(
                f"parse_json_list: Could not find a JSON array structure in text starting with: {text[:200]}..."
            )
            return []
        json_text = match.group(1)
        data = json.loads(json_text)
        items = []
        if isinstance(data, list):
            for item_dict in data:
                if (
                    isinstance(item_dict, dict)
                    and "name" in item_dict
                    and "description" in item_dict
                ):
                    items.append(
                        (str(item_dict["name"]), str(item_dict["description"]))
                    )
                else:
                    local_logger.warning(
                        f"parse_json_list: Skipping invalid item in JSON list: {item_dict}"
                    )
            return items[:max_items]
        else:
            local_logger.warning(
                f"parse_json_list: Parsed JSON is not a list. Type: {type(data)}. Data: {str(data)[:200]}..."
            )
            return []
    except json.JSONDecodeError as e:
        local_logger.error(
            f"parse_json_list: JSONDecodeError parsing text: {e}. Text snippet: {text[:500]}..."
        )
        return []
    except Exception as e:
        local_logger.error(
            f"parse_json_list: Unexpected error parsing JSON: {e}. Text snippet: {text[:500]}..."
        )
        return []


# --- Task Counter for Throttling ---
task_counter = {"count": 0}


def run_task_with_review(
    main_task: Task,
    review_task_fn: Callable[[Task], Task],
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Run a main task and its reviewer, returning their outputs with error handling.

    Returns a dict with keys: main, review, error (bool), error_msg (str or None).
    """
    try:
        review_task = review_task_fn(main_task)
        main_agent = main_task.agent
        review_agent = review_task.agent
        if main_agent is None:
            error_msg = "Main task is missing an agent."
            if logger:
                logger.error(error_msg)
            return {
                "main": "",
                "review": f"ERROR: {error_msg}",
                "error": True,
                "error_msg": error_msg,
            }
        if review_agent is None:
            error_msg = "Review task is missing an agent."
            if logger:
                logger.error(error_msg)
            return {
                "main": "",
                "review": f"ERROR: {error_msg}",
                "error": True,
                "error_msg": error_msg,
            }
        crew = Crew(
            agents=[main_agent, review_agent],
            tasks=[main_task, review_task],
            process=Process.sequential,
            verbose=bool(verbose),
        )
        crew.kickoff()
        task_counter["count"] += 2
        return {
            "main": (
                getattr(main_task.output, "raw_output", "")
                if hasattr(main_task, "output")
                else ""
            ),
            "review": (
                getattr(review_task.output, "raw_output", "")
                if hasattr(review_task, "output")
                else ""
            ),
            "error": False,
            "error_msg": None,
        }
    except Exception as e:
        if logger:
            logger.error(f"Task or review failed: {e}")
        return {"main": "", "review": f"ERROR: {e}", "error": True, "error_msg": str(e)}


def create_refinement_task(
    create_task_fn: Callable[[str, str], Task],
    name: str,
    desc: str,
    original_output: str,
    review_output: str,
    cycle: int,
    output_format_instruction: str,
) -> Task:
    return Task(
        description=(
            f"Refine your previous output for '{name}'.\n"
            f"Original description: {desc}\n"
            f"Your previous output:\n{original_output}\n"
            f"Peer review feedback:\n{review_output}\n"
            f"This is refinement cycle {cycle}. Please address the review feedback. "
            f"{output_format_instruction}"
        ),
        expected_output="A revised output in the specified format.",
        agent=create_task_fn(name, desc).agent,
    )


def run_task_with_review_and_refine(
    name: str,
    desc: str,
    create_task_fn: Callable[[str, str], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str], Task], str, str, str, str, int], Task
    ],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    logger: logging.Logger,
    verbose: int,
    review_cycles: int,
    parse_retries: int,
    max_items: int,
) -> Dict[str, Any]:
    """Run manager, reviewer, and optional refinement cycles. Returns dict with all outputs."""
    outputs = []
    current_output = None
    current_review = None
    error = False
    error_msg = None
    for cycle in range(review_cycles):
        # Manager or refinement task
        if cycle == 0:
            task = create_task_fn(name, desc)
        else:
            task = create_refine_fn(
                create_task_fn,
                name,
                desc,
                current_output if current_output is not None else "",
                current_review if current_review is not None else "",
                cycle,
            )
        # Retry on parse failure
        for attempt in range(parse_retries):
            try:
                review_task = create_review_fn(task)
                agents = [
                    agent
                    for agent in [task.agent, review_task.agent]
                    if agent is not None
                ]
                if len(agents) != 2:
                    raise ValueError(
                        "Both task.agent and review_task.agent must be non-None."
                    )
                crew = Crew(
                    agents=agents,
                    tasks=[task, review_task],
                    process=Process.sequential,
                    verbose=bool(verbose),
                )
                crew.kickoff()
                task_counter["count"] += 2
                main_output = ""
                parsed = []
                if hasattr(task, "output") and task.output is not None:
                    if (
                        hasattr(task.output, "pydantic")
                        and task.output.pydantic is not None
                        and isinstance(task.output.pydantic, ItemListOutput)
                    ):
                        logger.info(
                            f"Using task.output.pydantic for '{name}' (ItemListOutput)"
                        )
                        pydantic_model = task.output.pydantic
                        parsed = [
                            (item.name, item.description)
                            for item in pydantic_model.items
                        ]
                        main_output = pydantic_model.model_dump_json(indent=2)
                    elif (
                        hasattr(task.output, "raw")
                        and task.output.raw
                        and task.output.raw.strip()
                    ):
                        logger.warning(
                            f"Falling back to raw output for '{name}' (pydantic output not available or wrong type)"
                        )
                        main_output = task.output.raw
                        parsed = parse_json_list(main_output, max_items)
                    else:
                        logger.warning(
                            f"No usable output (pydantic or raw) found for '{name}'. task.output attributes: {vars(task.output) if task.output else 'None'}"
                        )
                else:
                    logger.warning(
                        f"Task '{name}' has no output object or output is None."
                    )
                logger.debug(
                    f"Content used for parsing for '{name}':\n'''{main_output}'''"
                )
                logger.debug(f"Parsed items for '{name}': {parsed}")
                review_output = (
                    getattr(review_task.output, "raw_output", "")
                    if hasattr(review_task, "output")
                    else ""
                )
                if not parsed and main_output.strip():
                    logger.warning(
                        f"Parsing produced no items for list-producing task '{name}' in cycle {cycle+1}, attempt {attempt+1}. Output was: {main_output[:200]}. Retrying..."
                    )
                    if attempt == parse_retries - 1:
                        error = True
                        error_msg = f"Parse failed after {parse_retries} attempts for '{name}'. Output could not be parsed into expected items."
                        logger.error(
                            f"Final output from '{name}' that failed structured parsing or item extraction:\n'''{main_output}'''"
                        )
                elif not parsed and not main_output.strip():
                    logger.warning(
                        f"No output produced by agent for list-producing task '{name}' in cycle {cycle+1}, attempt {attempt+1}."
                    )
                current_output = main_output
                current_review = review_output
                outputs.append(
                    {
                        "cycle": cycle + 1,
                        "main_output": main_output,
                        "review": review_output,
                    }
                )
                break
            except Exception as e:
                logger.error(f"Task or review failed: {e}")
                error = True
                error_msg = str(e)
                break
        if error:
            break
    return {
        "cycles": outputs,
        "final_output": current_output,
        "final_review": current_review,
        "error": error,
        "error_msg": error_msg,
    }


def orchestrate_level(
    items: List[Tuple[str, str]],
    create_task_fn: Callable[[str, str], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str], Task], str, str, str, str, int], Task
    ],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn: Optional[Callable],
    recursion_depth: int,
    max_depth: int,
    max_items: int,
    logger: logging.Logger,
    max_tasks: int,
    verbose: int,
    review_cycles: int,
    parse_retries: int,
) -> List[Dict[str, Any]]:
    """
    Generic recursive orchestration for modules/classes/functions.

    Args:
        items: List of (name, description) pairs for this level.
        create_task_fn: Function to create the main task.
        create_review_fn: Function to create the review task.
        parse_fn: Function to parse the next level's output.
        next_level_fn: Function to orchestrate the next level. Must accept (items, recursion_depth, logger, max_depth, max_items, max_tasks, verbose).
        recursion_depth: Current recursion depth.
        max_depth: Maximum allowed recursion depth.
        max_items: Maximum items per level.
        logger: Logger instance.
        max_tasks: Maximum total tasks allowed.
        verbose: Crew/agent verbosity level.
        review_cycles: Number of review/refinement cycles per task.
        parse_retries: Number of times to retry a task if output parsing fails.
    Returns:
        List[Dict[str, Any]]: Results for this level.

    Example:
        orchestrate_level(
            items, create_module_manager_task, create_module_review_task, parse_numbered_list, next_class_level,
            recursion_depth, max_depth, max_items, logger, max_tasks, verbose, review_cycles, parse_retries
        )
    """
    if recursion_depth > max_depth:
        logger.warning(
            f"Recursion depth {recursion_depth} exceeds max {max_depth}. Aborting deeper recursion."
        )
        return []
    results = []
    for name, desc in items[:max_items]:
        if task_counter["count"] >= max_tasks:
            logger.warning(f"Task cap reached ({max_tasks}). Halting further spawning.")
            break
        outputs = run_task_with_review_and_refine(
            name,
            desc,
            create_task_fn,
            create_review_fn,
            create_refine_fn,
            parse_fn,
            logger,
            verbose,
            review_cycles,
            parse_retries,
            max_items,
        )
        if outputs.get("error"):
            logger.error(f"Critical error in task '{name}': {outputs['error_msg']}")
            results.append(
                {
                    "name": name,
                    "description": desc,
                    "cycles": outputs["cycles"],
                    "final_output": outputs["final_output"],
                    "final_review": outputs["final_review"],
                    "children": [],
                    "error": True,
                    "error_msg": outputs["error_msg"],
                }
            )
            break
        next_items = parse_fn(outputs["final_output"], max_items)
        next_results = []
        if next_level_fn:
            next_results = next_level_fn(
                next_items,
                recursion_depth=recursion_depth + 1,
                logger=logger,
                max_depth=max_depth,
                max_items=max_items,
                max_tasks=max_tasks,
                verbose=verbose,
                review_cycles=review_cycles,
                parse_retries=parse_retries,
            )
        results.append(
            {
                "name": name,
                "description": desc,
                "cycles": outputs["cycles"],
                "final_output": outputs["final_output"],
                "final_review": outputs["final_review"],
                "children": next_results,
                "error": False,
                "error_msg": None,
            }
        )
    return results


def make_next_level_handler(
    create_task_fn, create_review_fn, create_refine_fn, parse_fn, next_level_fn
):
    """Factory for next-level orchestration handlers."""

    def handler(
        items,
        recursion_depth,
        logger,
        max_depth,
        max_items,
        max_tasks,
        verbose,
        review_cycles,
        parse_retries,
    ):
        return orchestrate_level(
            items,
            create_task_fn,
            create_review_fn,
            create_refine_fn,
            parse_fn,
            next_level_fn,
            recursion_depth,
            max_depth,
            max_items,
            logger,
            max_tasks,
            verbose,
            review_cycles,
            parse_retries,
        )

    return handler


def main():
    """Run the fully automated hierarchical CrewAI programming workflow with peer reviewers."""
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Hierarchical CrewAI programming team orchestrator."
    )
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
        "--max-depth", type=int, default=4, help="Maximum recursion depth (default: 4)"
    )
    parser.add_argument(
        "--max-items", type=int, default=5, help="Maximum items per level (default: 5)"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum total tasks allowed (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbosity level for Crew/agents and logging (0=INFO, 1=DEBUG, 2+=DEBUG+). Default: 0.",
    )
    parser.add_argument(
        "--review-cycles",
        type=int,
        default=1,
        help="Number of review/refinement cycles per task (default: 1)",
    )
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=1,
        help="Number of times to retry a task if output parsing fails (default: 1)",
    )
    args = parser.parse_args()

    project_path = args.project_path
    # Load project-specific configuration
    config_file = os.path.join(project_path, "config.yaml")
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Override args with config values
    requirements = (
        args.requirements
        or config.get("requirements")
        or "A command-line todo list application that allows users to add, remove, list, and mark tasks as complete. Data should be persisted between runs."
    )
    max_depth = config.get("max_depth", args.max_depth)
    max_items = config.get("max_items", args.max_items)
    max_tasks = config.get("max_tasks", args.max_tasks)
    review_cycles = config.get("review_cycles", args.review_cycles)
    parse_retries = config.get("parse_retries", args.parse_retries)
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)
    # Determine output directory and instantiate artifact store
    output_dir = os.path.join(project_path, "output")
    artifact_store = FileSystemArtifactStore(output_dir, logger)

    # Create scoped artifact tools for system-level access
    system_read_tool = ScopedGetArtifactTool(
        store=artifact_store,
        agent_role="System",
        allowed_read_prefixes=[""],
    )
    system_write_tool = ScopedSaveArtifactTool(
        store=artifact_store,
        agent_role="System",
        allowed_write_prefixes=[""],
    )

    logger.info("Starting the Fully Automated Programming Crew...")

    def handle_interrupt(signum, frame):
        logger.warning("Caught interrupt signal, shutting down gracefully.")
        exit(1)

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        signal.signal(signal.SIGTERM, handle_interrupt)
    except AttributeError:
        pass  # Not all systems have SIGTERM
    try:
        results = orchestrate_full_workflow(
            requirements,
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
            return
        logger.info(
            "System Architecture (final, after %d cycle(s)):\n%s",
            len(results.get("architecture_cycles", [])),
            results["architecture"],
        )
        logger.info("Architecture Review (final):\n%s", results["architecture_review"])
        for mod in results["modules"]:
            logger.info(
                "Module: %s\nDescription: %s\nFinal Review: %s\n(Cycles: %d)",
                mod["name"],
                mod["description"],
                mod["final_review"],
                len(mod["cycles"]),
            )
            for cls in mod["children"]:
                logger.info(
                    "  Class: %s\n  Description: %s\n  Final Review: %s\n  (Cycles: %d)",
                    cls["name"],
                    cls["description"],
                    cls["final_review"],
                    len(cls["cycles"]),
                )
                for fn in cls["children"]:
                    logger.info(
                        "    Function: %s\n    Description: %s\n    Implementation (final):\n%s\n    Final Review: %s\n    (Cycles: %d)",
                        fn["name"],
                        fn["description"],
                        fn["final_output"],
                        fn["final_review"],
                        len(fn["cycles"]),
                    )
        print("\n===== JSON SUMMARY =====\n")
        print(json.dumps(results, indent=2))
    except KeyboardInterrupt:
        logger.warning("Caught KeyboardInterrupt, shutting down gracefully.")
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")


if __name__ == "__main__":
    main()

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

load_dotenv()

# --- Agent Definitions ---


def create_architect_agent():
    """Create the Systems Architect agent."""
    return Agent(
        role="Systems Architect",
        goal="Design a robust, modular software system for the given requirements.",
        backstory=(
            "You are a highly experienced systems architect. You break down complex requirements "
            "into high-level modules, ensuring scalability and maintainability. You delegate module "
            "design to module managers."
        ),
        verbose=True,
        allow_delegation=True,
    )


def create_architect_reviewer_agent():
    """Create the Architect Reviewer agent."""
    return Agent(
        role="Architect Reviewer",
        goal="Critique and enhance the system architecture for robustness and clarity.",
        backstory=(
            "You are a peer systems architect, known for your critical eye and ability to spot design flaws. "
            "You review architectural plans and suggest improvements."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_module_manager_agent():
    """Create a Module Manager agent."""
    return Agent(
        role="Module Manager",
        goal="Design a module, define its classes, and delegate class design to class managers.",
        backstory=(
            "You are responsible for the detailed design of a software module. You identify the "
            "necessary classes and delegate their design to class managers."
        ),
        verbose=True,
        allow_delegation=True,
    )


def create_module_reviewer_agent():
    """Create a Module Reviewer agent."""
    return Agent(
        role="Module Reviewer",
        goal="Critique and enhance the module design for cohesion and completeness.",
        backstory=(
            "You are a peer module designer, skilled at reviewing module boundaries and class selection. "
            "You suggest improvements and spot missing elements."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_class_manager_agent():
    """Create a Class Manager agent."""
    return Agent(
        role="Class Manager",
        goal="Design a class, define its functions, and delegate function implementation to function managers.",
        backstory=(
            "You are responsible for the design of a class within a module. You identify the necessary "
            "functions/methods and delegate their implementation to function managers."
        ),
        verbose=True,
        allow_delegation=True,
    )


def create_class_reviewer_agent():
    """Create a Class Reviewer agent."""
    return Agent(
        role="Class Reviewer",
        goal="Critique and enhance the class design for clarity and extensibility.",
        backstory=(
            "You are a peer class designer, skilled at reviewing class responsibilities and method selection. "
            "You suggest improvements and spot missing or redundant methods."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_function_manager_agent():
    """Create a Function Manager agent."""
    return Agent(
        role="Function Manager",
        goal="Implement a function or method as specified by the class manager.",
        backstory=(
            "You are responsible for implementing a single function or method, ensuring it meets the "
            "requirements and integrates with the class design."
        ),
        verbose=True,
        allow_delegation=False,
    )


def create_function_reviewer_agent():
    """Create a Function Reviewer agent."""
    return Agent(
        role="Function Reviewer",
        goal="Critique and enhance the function implementation for correctness and style.",
        backstory=(
            "You are a peer function implementer, skilled at reviewing code for bugs, clarity, and efficiency. "
            "You suggest improvements and spot potential issues."
        ),
        verbose=True,
        allow_delegation=False,
    )


# --- Task Definitions ---


def create_architect_task(requirements):
    """Create the Systems Architect's task."""
    return Task(
        description=(
            f"Analyze the following requirements and design a modular system. "
            f"List the modules needed, and for each, briefly describe its purpose. "
            f"Output MUST be a JSON array of objects, each with 'name' and 'description' fields, e.g.\n"
            f"[{{'name': 'ModuleName', 'description': 'Purpose of the module'}}, ...] (no extra text).\n"
            f"Requirements: {requirements}"
        ),
        expected_output="A JSON array of objects, each with 'name' and 'description' fields.",
        agent=create_architect_agent(),
    )


def create_architect_review_task(arch_task):
    """Create the Architect Reviewer's task."""
    return Task(
        description=(
            "Review the proposed system architecture. Critique its modularity, scalability, and clarity. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the architecture.",
        agent=create_architect_reviewer_agent(),
        context=[arch_task],
    )


def create_module_manager_task(module_name, module_description):
    """Create a Module Manager's task."""
    return Task(
        description=(
            f"Design the module '{module_name}': {module_description}. "
            f"List the classes needed, and for each, briefly describe its purpose. "
            f"Output MUST be a JSON array of objects, each with 'name' and 'description' fields, e.g.\n"
            f"[{{'name': 'ClassName', 'description': 'Purpose of the class'}}, ...] (no extra text)."
        ),
        expected_output="A JSON array of objects, each with 'name' and 'description' fields.",
        agent=create_module_manager_agent(),
    )


def create_module_review_task(mod_task):
    """Create a Module Reviewer task."""
    return Task(
        description=(
            "Review the proposed module design. Critique its cohesion, completeness, and class selection. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the module design.",
        agent=create_module_reviewer_agent(),
        context=[mod_task],
    )


def create_class_manager_task(class_name, class_description):
    """Create a Class Manager's task."""
    return Task(
        description=(
            f"Design the class '{class_name}': {class_description}. "
            f"List the functions/methods needed, and for each, briefly describe its purpose. "
            f"Output MUST be a JSON array of objects, each with 'name' and 'description' fields, e.g.\n"
            f"[{{'name': 'FunctionName', 'description': 'Purpose of the function'}}, ...] (no extra text)."
        ),
        expected_output="A JSON array of objects, each with 'name' and 'description' fields.",
        agent=create_class_manager_agent(),
    )


def create_class_review_task(cls_task):
    """Create a Class Reviewer task."""
    return Task(
        description=(
            "Review the proposed class design. Critique its clarity, extensibility, and method selection. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the class design.",
        agent=create_class_reviewer_agent(),
        context=[cls_task],
    )


def create_function_manager_task(function_name, function_description):
    """Create a Function Manager's task."""
    return Task(
        description=(
            f"Implement the function/method '{function_name}': {function_description}. "
            f"Provide the full code implementation with a docstring."
        ),
        expected_output="The complete code for the function/method, with a docstring.",
        agent=create_function_manager_agent(),
    )


def create_function_review_task(fn_task):
    """Create a Function Reviewer task."""
    return Task(
        description=(
            "Review the function implementation. Critique its correctness, clarity, and efficiency. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the function implementation.",
        agent=create_function_reviewer_agent(),
        context=[fn_task],
    )


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
        return []
    try:
        # LLMs sometimes wrap JSON in markdown ```json ... ```
        match = re.search(r"(\[[\s\S]*\])", text)
        if not match:
            local_logger.warning(
                f"Could not find a JSON array structure in text: {text[:200]}..."
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
                        f"Skipping invalid item in JSON list: {item_dict}"
                    )
            return items[:max_items]
        else:
            local_logger.warning(f"Parsed JSON is not a list: {type(data)}")
            return []
    except json.JSONDecodeError as e:
        local_logger.error(
            f"JSONDecodeError parsing text: {e}\nText was: {text[:500]}..."
        )
        return []
    except Exception as e:
        local_logger.error(
            f"Unexpected error parsing JSON: {e}\nText was: {text[:500]}..."
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
                main_output = (
                    getattr(task.output, "raw_output", "")
                    if hasattr(task, "output")
                    else ""
                )
                logger.debug(
                    f"Attempting to parse main_output for '{name}' (cycle {cycle+1}, attempt {attempt+1}):\n'''{main_output}'''"
                )
                review_output = (
                    getattr(review_task.output, "raw_output", "")
                    if hasattr(review_task, "output")
                    else ""
                )
                parsed = parse_fn(main_output, max_items)
                if not parsed:
                    logger.warning(
                        f"Parse failed for output in cycle {cycle+1}, attempt {attempt+1}. Retrying..."
                    )
                    if attempt == parse_retries - 1:
                        error = True
                        error_msg = f"Parse failed after {parse_retries} attempts."
                        break
                    continue
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


def orchestrate_full_workflow(
    requirements: str,
    max_depth: int = 4,
    max_items: int = 5,
    max_tasks: int = 100,
    verbose: int = 0,
    logger: Optional[logging.Logger] = None,
    review_cycles: int = 1,
    parse_retries: int = 1,
) -> Dict[str, Any]:
    """Orchestrate the full hierarchical workflow with peer review at each level.

    Args:
        requirements (str): The high-level requirements.
        max_depth (int): Maximum recursion depth.
        max_items (int): Maximum items per level.
        max_tasks (int): Maximum total tasks allowed.
        verbose (int): Crew/agent verbosity level.
        logger (logging.Logger, optional): Logger instance.
        review_cycles (int): Number of review/refinement cycles per task.
        parse_retries (int): Number of times to retry a task if output parsing fails.
    Returns:
        Dict[str, Any]: Nested results for all levels.
    """
    logger = logger or logging.getLogger(__name__)
    task_counter["count"] = 0
    # Architect-level refinement
    json_output_instruction = (
        "Output MUST be a JSON array of objects, each with 'name' and 'description' fields, "
        "e.g. [{'name': 'ModuleName', 'description': 'Purpose of the module'}, ...] (no extra text)."
    )
    architect_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, json_output_instruction
    )
    arch_outputs = run_task_with_review_and_refine(
        name="System Architecture",
        desc=requirements,
        create_task_fn=lambda _, requirements_arg: create_architect_task(
            requirements_arg
        ),
        create_review_fn=create_architect_review_task,
        create_refine_fn=architect_create_refine_fn,
        parse_fn=lambda text, max_items: parse_json_list(
            text, max_items, logger=logger
        ),
        logger=logger,
        verbose=verbose,
        review_cycles=review_cycles,
        parse_retries=parse_retries,
        max_items=max_items,
    )
    if arch_outputs.get("error"):
        logger.error(f"Critical error in architecture: {arch_outputs['error_msg']}")
        return {
            "architecture": arch_outputs["final_output"],
            "architecture_review": arch_outputs["final_review"],
            "modules": [],
            "error": True,
            "error_msg": arch_outputs["error_msg"],
        }
    modules = parse_json_list(arch_outputs["final_output"], max_items, logger=logger)
    numbered_list_instruction = "Output in the required format: Each item MUST be in the format '1. Name: Description' (one per line, no extra text)."
    default_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, numbered_list_instruction
    )
    next_function_level = make_next_level_handler(
        create_function_manager_task,
        create_function_review_task,
        default_create_refine_fn,
        lambda x, _: [],
        None,
    )
    next_class_level = make_next_level_handler(
        create_class_manager_task,
        create_class_review_task,
        default_create_refine_fn,
        parse_json_list,
        next_function_level,
    )
    next_module_level = make_next_level_handler(
        create_module_manager_task,
        create_module_review_task,
        default_create_refine_fn,
        parse_json_list,
        next_class_level,
    )
    modules_result = next_module_level(
        modules,
        1,
        logger,
        max_depth,
        max_items,
        max_tasks,
        verbose,
        review_cycles,
        parse_retries,
    )
    return {
        "architecture": arch_outputs["final_output"],
        "architecture_review": arch_outputs["final_review"],
        "architecture_cycles": arch_outputs["cycles"],
        "modules": modules_result,
        "error": False,
        "error_msg": None,
    }


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_api_keys():
    """Ensure required API keys are set for CrewAI agents."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment.")
    os.environ["OPENAI_API_KEY"] = openai_key  # Explicitly set for downstream consumers
    # Add other keys as needed
    # os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")


def main():
    """Run the fully automated hierarchical CrewAI programming workflow with peer reviewers."""
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(
        description="Hierarchical CrewAI programming team orchestrator."
    )
    parser.add_argument(
        "--requirements",
        type=str,
        required=False,
        help="High-level requirements for the system. (Default: a todo-list app)",
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
    ensure_api_keys()
    requirements = args.requirements or (
        "A command-line todo list application that allows users to add, remove, "
        "list, and mark tasks as complete. Data should be persisted between runs."
    )
    if args.verbose > 0:
        logging.getLogger().setLevel(logging.DEBUG)
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
            max_depth=args.max_depth,
            max_items=args.max_items,
            max_tasks=args.max_tasks,
            verbose=args.verbose,
            logger=logger,
            review_cycles=args.review_cycles,
            parse_retries=args.parse_retries,
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

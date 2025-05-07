import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
from crewai import Task, Crew, Process
from convoke.agents import ItemListOutput
from convoke.utils import lint_python_code
from convoke.store import FileSystemArtifactStore
from convoke.tools import ScopedGetArtifactTool, ScopedSaveArtifactTool
import json
from convoke.project import write_project_scaffolding, write_tree_visualization
from convoke.agents import (
    create_architect_task,
    create_architect_review_task,
    create_module_manager_task,
    create_module_review_task,
    create_class_manager_task,
    create_class_review_task,
    create_function_manager_task,
    create_function_review_task,
    ItemListOutput,
)
from convoke.parsers import parse_json_list
from convoke.project import write_project_scaffolding, write_tree_visualization

__all__ = [
    "parse_numbered_list",
    "parse_json_list",
    "orchestrate_level",
    "orchestrate_full_workflow",
]


def parse_numbered_list(text: str, max_items: int = 10) -> List[Tuple[str, str]]:
    if not isinstance(text, str):
        return []
    try:
        lines = text.splitlines()
    except Exception:
        return []
    items = []
    current_name = None
    current_desc_lines = []
    import re

    for line in lines:
        m = re.match(r"^\s*(\d+|[-*])\.?\s+([^:]+):?\s*(.*)$", line)
        if m:
            if current_name is not None:
                items.append((current_name, "\n".join(current_desc_lines).strip()))
            current_name = m.group(2).strip()
            desc = m.group(3).strip()
            current_desc_lines = [desc] if desc else []
        elif current_name is not None and not re.match(r"^\s*(\d+|[-*])\.?\s+", line):
            current_desc_lines.append(line)
    if current_name is not None:
        items.append((current_name, "\n".join(current_desc_lines).strip()))
    return items[:max_items]


def parse_json_list(
    text: str, max_items: int = 10, logger: Optional[logging.Logger] = None
) -> List[Tuple[str, str]]:
    import re

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


task_counter = {"count": 0}


def run_task_with_review(
    main_task: Task,
    review_task_fn: Callable[[Task], Task],
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
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


def orchestrate_level(
    items: List[Tuple[str, str]],
    create_task_fn: Callable[[str, str], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str], Task], str, str, str, str, int], Task
    ],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn: Optional[
        Callable[
            [List[Tuple[str, str]], int, logging.Logger, int, int, int, int, int, int],
            List[Dict[str, Any]],
        ]
    ],
    recursion_depth: int,
    max_depth: int,
    max_items: int,
    logger: logging.Logger,
    max_tasks: int,
    verbose: int,
    review_cycles: int,
    parse_retries: int,
) -> List[Dict[str, Any]]:
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
                    "cycles": outputs.get("cycles", []),
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
                "cycles": outputs.get("cycles", []),
                "final_output": outputs["final_output"],
                "final_review": outputs["final_review"],
                "children": next_results,
                "error": False,
                "error_msg": None,
            }
        )
    return results


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
    outputs = []
    current_output = None
    current_review = None
    error = False
    error_msg = None
    for cycle in range(review_cycles):
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
                    # If output is a JSON list (possibly empty), accept it without error
                    stripped = main_output.strip()
                    if not (stripped.startswith("[") and stripped.endswith("]")):
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


def create_refinement_task(
    create_task_fn: Callable[[str, str], Task],
    name: str,
    desc: str,
    original_output: str,
    review_output: str,
    cycle: int,
) -> Task:
    """
    Stub refinement: simply re-run the original task for further cycles.
    """
    # In deeper cycles, we ignore feedback and re-invoke the task creator.
    return create_task_fn(name, desc)


import re


def extract_json_from_output(text: str) -> str:
    match = re.search(r"(\[[\s\S]*\])", text)
    if match:
        return match.group(1)
    return ""


def make_next_level_handler(
    create_task_fn: Callable[[str, str], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[[Callable[[str, str], Task], str, str, str, str, int], Task],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn: Optional[Callable],
) -> Callable[..., List[Dict[str, Any]]]:
    def handler(
        items: List[Tuple[str, str]],
        recursion_depth: int,
        logger: logging.Logger,
        max_depth: int,
        max_items: int,
        max_tasks: int,
        verbose: int,
        review_cycles: int,
        parse_retries: int,
    ) -> List[Dict[str, Any]]:
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
    max_depth: int,
    max_items: int,
    max_tasks: int,
    verbose: int,
    logger: logging.Logger,
    review_cycles: int,
    parse_retries: int,
    output_dir: str,
    artifact_store: FileSystemArtifactStore,
    get_tool: Any,
    save_tool: Any,
) -> Dict[str, Any]:
    """
    Main entry point for the full workflow orchestration. Coordinates all levels and writes outputs.
    """
    logger = logger or logging.getLogger(__name__)
    # Handle empty requirements: no architecture or modules
    if not requirements or not requirements.strip():
        return {
            "architecture": "",
            "architecture_review": "",
            "modules": [],
            "error": False,
            "error_msg": None,
        }

    # 1. System Architecture with review and refinement
    # wrap architect fns to match expected signatures
    architect_ctf: Callable[[str, str], Task] = lambda name, desc: create_architect_task(desc)
    architect_rtf: Callable[[Task], Task] = lambda t: create_architect_review_task(t)
    arch_outputs = run_task_with_review_and_refine(
        "System Architecture",
        requirements,
        architect_ctf,
        architect_rtf,
        create_refinement_task,
        parse_json_list,
        logger,
        verbose,
        review_cycles,
        parse_retries,
        max_items,
    )
    if arch_outputs.get("error"):
        # On architect error, return minimal structure
        return {
            "architecture": "",
            "architecture_review": "",
            "modules": [],
            "error": True,
            "error_msg": arch_outputs.get("error_msg"),
        }
    architecture_str = arch_outputs.get("final_output", "")
    architecture_review = arch_outputs.get("final_review", "")

    # Parse modules list from architect output
    modules = parse_json_list(extract_json_from_output(architecture_str), max_items, logger)

    # Set up recursion handlers for classes and functions
    function_handler = make_next_level_handler(
        create_function_manager_task,
        create_function_review_task,
        create_refinement_task,
        parse_json_list,
        None,
    )
    class_handler = make_next_level_handler(
        create_class_manager_task,
        create_class_review_task,
        create_refinement_task,
        parse_json_list,
        function_handler,
    )

    # 2. Recursive orchestration starting at module level
    module_results = orchestrate_level(
        modules,
        create_module_manager_task,
        create_module_review_task,
        create_refinement_task,
        parse_json_list,
        class_handler,
        recursion_depth=0,
        max_depth=max_depth,
        max_items=max_items,
        logger=logger,
        max_tasks=max_tasks,
        verbose=verbose,
        review_cycles=review_cycles,
        parse_retries=parse_retries,
    )

    # Write project scaffolding and visualization
    write_project_scaffolding(output_dir, architecture_str, modules)
    write_tree_visualization(output_dir, module_results)

    return {
        "architecture": architecture_str,
        "architecture_review": architecture_review,
        "modules": module_results,
        "error": False,
        "error_msg": None,
    }

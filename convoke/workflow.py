import logging
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
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
import os
from convoke.utils import (
    ensure_dir_exists,
    write_json_file,
    write_text_file,
    lint_python_code,
)
from convoke.agents import create_test_developer_task, create_test_review_task

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
    create_task_fn: Callable[[str, str, Optional[list]], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str, Optional[list]], Task], str, str, str, str, int], Task
    ],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn_or_depth: Optional[Union[Callable, int]],
    max_depth: int,
    max_items: int,
    logger: logging.Logger,
    max_tasks: int,
    verbose: int,
    review_cycles: int,
    parse_retries: int,
    artifact_store: FileSystemArtifactStore = None,
    current_scope_path: str = "",
) -> List[Dict[str, Any]]:
    # Handle both legacy usage and new usage
    recursion_depth = 0
    next_level_fn = None
    if isinstance(next_level_fn_or_depth, int):
        recursion_depth = next_level_fn_or_depth
    else:
        recursion_depth = 0  # Default if not provided
        next_level_fn = next_level_fn_or_depth

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
        # Compute scope for this item
        item_full_scope_path = os.path.join(current_scope_path, name.replace(" ", "_"))
        # Determine role (optional: infer from create_task_fn or pass as param)
        role = getattr(create_task_fn, "role", "Agent")
        # Setup scoped tools
        # Allow reading from current scope and parent scope
        allowed_read_prefixes = []
        if current_scope_path:
            allowed_read_prefixes.append(current_scope_path)
            parent_scope = os.path.dirname(current_scope_path)
            if parent_scope and parent_scope != ".":
                allowed_read_prefixes.append(parent_scope)
        else:
            allowed_read_prefixes.append("")  # root
        allowed_read_prefixes = list(set(allowed_read_prefixes))
        allowed_write_prefixes = [item_full_scope_path]
        tools = []
        if artifact_store:
            tools = [
                ScopedGetArtifactTool(
                    store=artifact_store,
                    agent_role=role,
                    allowed_read_prefixes=allowed_read_prefixes,
                ),
                ScopedSaveArtifactTool(
                    store=artifact_store,
                    agent_role=role,
                    allowed_write_prefixes=allowed_write_prefixes,
                ),
            ]
        # Pass tools to agent creation for both main and review tasks
        outputs = run_task_with_review_and_refine(
            name,
            desc,
            lambda n, d: create_task_fn(n, d, tools),
            lambda t: create_review_fn(t, tools),
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
                    # 'cycles' may be missing in test patches
                    "cycles": outputs.get("cycles", []),
                    "final_output": outputs.get("final_output"),
                    "final_review": outputs.get("final_review"),
                    "children": [],
                    "error": True,
                    "error_msg": outputs.get("error_msg"),
                }
            )
            break
        # Use parsed_list directly (do not re-parse)
        next_items = outputs.get("parsed_list", [])
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
                artifact_store=artifact_store,
                current_scope_path=item_full_scope_path,
            )
        results.append(
            {
                "name": name,
                "description": desc,
                # 'cycles' may be missing in test patches
                "cycles": outputs.get("cycles", []),
                "final_output": outputs.get("final_output"),
                "final_review": outputs.get("final_review"),
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
                    getattr(review_task.output, "raw", "")
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
                        "parsed_list": parsed,
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
        "parsed_list": current_output and parsed or [],
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
    output_format_instruction: str,
) -> Task:
    """
    Create a refinement task that includes previous output, peer feedback, and output format instructions.
    """
    base_task = create_task_fn(name, desc)
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
        agent=base_task.agent,
        context=[base_task],
    )


import re


def extract_json_from_output(text: str) -> str:
    match = re.search(r"(\[[\s\S]*\])", text)
    if match:
        return match.group(1)
    return ""


def extract_items_from_pydantic_output(
    task_output, max_items: int = 10
) -> List[Tuple[str, str]]:
    if hasattr(task_output, "pydantic") and isinstance(
        task_output.pydantic, ItemListOutput
    ):
        return [(item.name, item.description) for item in task_output.pydantic.items][
            :max_items
        ]
    return []


# Fix the wrapped_next function in make_next_level_handler_with_tools to match signature
def make_next_level_handler(
    create_task_fn: Callable[[str, str], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str], Task], str, str, str, str, int], Task
    ],
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


# Fix the wrapped_next function in make_next_level_handler_with_tools to match signature
def make_next_level_handler_with_tools(
    orig_fn: Callable[..., Task],
    orig_review_fn: Callable[..., Task],
    role: str,
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn: Optional[Callable],
) -> Callable[..., List[Dict[str, Any]]]:
    # This handler is now redundant since orchestrate_level handles tool injection.
    # Use the simpler make_next_level_handler instead.
    return make_next_level_handler(
        orig_fn,
        orig_review_fn,
        create_refinement_task,
        parse_fn,
        next_level_fn,
    )


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
    architect_ctf: Callable[[str, str], Task] = (
        lambda name, desc: create_architect_task(desc)
    )
    architect_rtf: Callable[[Task], Task] = lambda t: create_architect_review_task(t)
    arch_outputs = run_task_with_review_and_refine(
        "System Architecture",
        requirements,
        architect_ctf,
        architect_rtf,
        extract_items_from_pydantic_output,  # Use pydantic extractor for architect
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
    modules = parse_json_list(
        extract_json_from_output(architecture_str), max_items, logger
    )

    # Internal helpers for wrapping task and review functions
    def _wrap_task(orig_fn, name, desc, role, parent_scope):
        task = orig_fn(name, desc)
        slug = name.replace(" ", "_")
        scope_path = os.path.join(parent_scope, slug).lstrip("/")
        get_scoped = ScopedGetArtifactTool(
            store=artifact_store, agent_role=role, allowed_read_prefixes=[parent_scope]
        )
        save_scoped = ScopedSaveArtifactTool(
            store=artifact_store, agent_role=role, allowed_write_prefixes=[scope_path]
        )
        task.agent.tools = [get_scoped, save_scoped]
        return task

    def _wrap_review(orig_review_fn, task_obj):
        review_task = orig_review_fn(task_obj)
        review_task.agent.tools = getattr(task_obj.agent, "tools", [])
        return review_task

    # Output format instructions for each level
    json_output_instruction = (
        "Output MUST be a JSON array of objects, each with 'name' and 'description' fields, "
        "e.g. [{'name': 'ModuleName', 'description': 'Purpose of the module'}, ...] (no extra text, no explanation)."
    )
    numbered_list_instruction = "Output in the required format: Each item MUST be in the format '1. Name: Description' (one per line, no extra text)."
    function_code_instruction = "Provide the full code implementation with a docstring, ensuring it meets the original requirements and addresses the review feedback."

    architect_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, json_output_instruction
    )
    module_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, json_output_instruction
    )
    class_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, json_output_instruction
    )
    function_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, function_code_instruction
    )

    # Build handlers: functions have no children, classes call functions, modules call classes
    function_handler = make_next_level_handler_with_tools(
        create_function_manager_task,
        create_function_review_task,
        "Function Manager",
        parse_json_list,
        None,
    )
    class_handler = make_next_level_handler_with_tools(
        create_class_manager_task,
        create_class_review_task,
        "Class Manager",
        parse_json_list,
        function_handler,
    )
    module_handler = make_next_level_handler_with_tools(
        create_module_manager_task,
        create_module_review_task,
        "Module Manager",
        parse_json_list,
        class_handler,
    )

    # 2. Recursive orchestration starting at module level
    module_results = module_handler(
        modules,
        "",  # top-level scope
        recursion_depth=0,
        max_depth=max_depth,
        max_items=max_items,
        logger=logger,
        max_tasks=max_tasks,
        verbose=verbose,
        review_cycles=review_cycles,
        parse_retries=parse_retries,
        artifact_store=artifact_store,
        current_scope_path="",
    )

    # Save individual artifacts (designs, reviews, code, tests)
    try:
        ensure_dir_exists(output_dir)
    except Exception as e:
        logger.error(f"Could not create output directory: {e}")
    for mod in module_results:
        # Save module artifacts
        try:
            mod_dir = os.path.join(output_dir, mod["name"].replace(" ", "_"))
            ensure_dir_exists(mod_dir)
            try:
                write_json_file(
                    os.path.join(mod_dir, "module_design.json"),
                    {
                        "name": mod["name"],
                        "description": mod["description"],
                        "cycles": mod["cycles"],
                        "final_output": mod["final_output"],
                    },
                )
            except Exception as e:
                logger.error(
                    f"Error saving module_design.json for '{mod['name']}']: {e}"
                )
            try:
                write_text_file(
                    os.path.join(mod_dir, "module_review.txt"),
                    mod["final_review"] or "",
                )
            except Exception as e:
                logger.error(
                    f"Error saving module_review.txt for '{mod['name']}']: {e}"
                )
        except Exception as e:
            logger.error(f"Error preparing module directory for '{mod['name']}']: {e}")
        for cls in mod.get("children", []):
            # Save class artifacts
            try:
                cls_dir = os.path.join(mod_dir, cls["name"].replace(" ", "_"))
                ensure_dir_exists(cls_dir)
                try:
                    write_json_file(
                        os.path.join(cls_dir, "class_design.json"),
                        {
                            "name": cls["name"],
                            "description": cls["description"],
                            "cycles": cls["cycles"],
                            "final_output": cls["final_output"],
                        },
                    )
                except Exception as e:
                    logger.error(
                        f"Error saving class_design.json for '{cls['name']}']: {e}"
                    )
                try:
                    write_text_file(
                        os.path.join(cls_dir, "class_review.txt"),
                        cls["final_review"] or "",
                    )
                except Exception as e:
                    logger.error(
                        f"Error saving class_review.txt for '{cls['name']}']: {e}"
                    )
            except Exception as e:
                logger.error(
                    f"Error preparing class directory for '{cls['name']}']: {e}"
                )
            for fn in cls.get("children", []):
                try:
                    fn_file = os.path.join(
                        cls_dir, f"{fn['name'].replace(' ', '_')}.py"
                    )
                    try:
                        write_text_file(fn_file, fn["final_output"] or "")
                    except Exception as e:
                        logger.error(
                            f"Error saving function code for '{fn['name']}']: {e}"
                        )
                    try:
                        write_text_file(
                            os.path.join(
                                cls_dir, f"{fn['name'].replace(' ', '_')}_review.txt"
                            ),
                            fn["final_review"] or "",
                        )
                    except Exception as e:
                        logger.error(
                            f"Error saving function review for '{fn['name']}']: {e}"
                        )
                    if fn["final_output"]:
                        # Lint code
                        try:
                            lint_result = lint_python_code(fn["final_output"])
                            write_text_file(
                                os.path.join(
                                    cls_dir, f"{fn['name'].replace(' ', '_')}_lint.txt"
                                ),
                                lint_result,
                            )
                        except Exception as le:
                            logger.error(
                                f"Linting failed for function '{fn['name']}']: {le}"
                            )
                        # Generate and review tests
                        try:
                            test_task = create_test_developer_task(
                                fn["name"], fn["final_output"]
                            )
                            test_review_task = create_test_review_task(test_task)
                            test_crew = Crew(
                                agents=[test_task.agent, test_review_task.agent],
                                tasks=[test_task, test_review_task],
                                process=Process.sequential,
                                verbose=False,
                            )
                            test_crew.kickoff()
                            test_code = getattr(test_task.output, "raw", "")
                            test_review = getattr(test_review_task.output, "raw", "")
                            try:
                                write_text_file(
                                    os.path.join(
                                        cls_dir,
                                        f"test_{fn['name'].replace(' ', '_')}.py",
                                    ),
                                    test_code,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error saving test code for '{fn['name']}']: {e}"
                                )
                            try:
                                write_text_file(
                                    os.path.join(
                                        cls_dir,
                                        f"test_{fn['name'].replace(' ', '_')}_review.txt",
                                    ),
                                    test_review,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error saving test review for '{fn['name']}']: {e}"
                                )
                        except Exception as te:
                            logger.error(
                                f"Test generation failed for function '{fn['name']}']: {te}"
                            )
                except Exception as e:
                    logger.error(
                        f"Error preparing function artifacts for '{fn['name']}']: {e}"
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

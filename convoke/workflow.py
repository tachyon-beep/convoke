import logging
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from crewai import Task, Crew, Process
from convoke.agents import ItemListOutput
from convoke.utils import lint_python_code
from convoke.store import FileSystemArtifactStore
from convoke.tools import scoped_get_artifact, scoped_save_artifact
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
from convoke.parsers import parse_json_list, extract_items_from_pydantic_output
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
    "parse_json_list",
    "orchestrate_level",
    "orchestrate_full_workflow",
]


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
        if (main_agent is None) or (review_agent is None):
            error_msg = "Main task or review task is missing an agent."
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
    create_task_fn: Callable[[str, str, Optional[List[Any]]], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str, Optional[List[Any]]], Task], str, str, str, str, int], Task
    ],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn_or_depth: Optional[Union[Callable[..., List[Dict[str, Any]]], int]],
    max_depth: int,
    max_items: int,
    logger: logging.Logger,
    max_tasks: int,
    verbose: int,
    review_cycles: int,
    parse_retries: int,
    artifact_store: Optional[FileSystemArtifactStore] = None,
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

    # Ensure we have a valid artifact store
    if artifact_store is None:
        logger.warning(
            "No artifact store provided to orchestrate_level, creating default store"
        )
        artifact_store = FileSystemArtifactStore(
            base_path="./artifacts", logger=logger, agent_role="orchestrator"
        )

    results = []
    for name, desc in items[:max_items]:
        if task_counter["count"] >= max_tasks:
            logger.warning(f"Task cap reached ({max_tasks}). Halting further spawning.")
            break

        # Compute scope for this item
        item_full_scope_path = os.path.join(current_scope_path, name.replace(" ", "_"))

        # Determine role (optional: infer from create_task_fn or pass as param)
        role = getattr(create_task_fn, "role", "Agent")

        # Create a scoped child store for this item
        item_store = artifact_store.create_child_store(
            scope_path=item_full_scope_path, agent_role=role
        )

        # Create tools that use the scoped store
        tools = [
            scoped_save_artifact,
            scoped_get_artifact,
        ]

        # Pass tools to agent creation for both main and review tasks
        outputs = run_task_with_review_and_refine(
            name,
            desc,
            lambda n, d: create_task_fn(n, d, tools),
            lambda t: create_review_fn(t),
            lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
                ctf,
                n,
                d,
                oo,
                ro,
                cyc,
                "Output MUST be a JSON array of objects with 'name' and 'description'.",
            ),
            parse_fn,
            logger,
            verbose,
            review_cycles,
            parse_retries,
            max_items,
            scoped_store=item_store,  # Pass the scoped store to run_task_with_review_and_refine
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
                artifact_store=item_store,  # Pass the item's store to the next level
                current_scope_path=item_full_scope_path,
            )

        results.append(
            {
                "name": name,
                "description": desc,
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
    scoped_store: Optional[FileSystemArtifactStore] = None,
) -> Dict[str, Any]:
    outputs = []
    current_output = None
    current_review = None
    error = False
    error_msg = None

    for cycle in range(review_cycles):
        if cycle == 0:
            # Pass tool_params to the task creation function
            task = create_task_fn(name, desc)

            # Attach store to the agent's tools if needed
            if (
                scoped_store
                and hasattr(task, "agent")
                and task.agent
                and hasattr(task.agent, "tools")
            ):
                for tool in task.agent.tools:
                    # Set the store parameter in the tool's parameters directly
                    if hasattr(tool, "params"):
                        tool.params["store"] = scoped_store
        else:
            task = create_refine_fn(
                create_task_fn,
                name,
                desc,
                current_output if current_output is not None else "",
                current_review if current_review is not None else "",
                cycle,
                "Output MUST be a JSON array of objects with 'name' and 'description'.",
            )

            # Attach store to the agent's tools if needed
            if (
                scoped_store
                and hasattr(task, "agent")
                and task.agent
                and hasattr(task.agent, "tools")
            ):
                for tool in task.agent.tools:
                    # Set the store parameter in the tool's parameters directly
                    if hasattr(tool, "params"):
                        tool.params["store"] = scoped_store

        # Also modify the review task to include the store
        review_task = create_review_fn(task)
        if (
            scoped_store
            and hasattr(review_task, "agent")
            and review_task.agent
            and hasattr(review_task.agent, "tools")
        ):
            for tool in review_task.agent.tools:
                if hasattr(tool, "params"):
                    tool.params["store"] = scoped_store

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


# Updated `make_next_level_handler` to include `artifact_store` and `current_scope_path` in the handler's signature and logic.
def make_next_level_handler(
    create_task_fn: Callable[[str, str, Optional[List[Any]]], Task],
    create_review_fn: Callable[[Task], Task],
    create_refine_fn: Callable[
        [Callable[[str, str, Optional[List[Any]]], Task], str, str, str, str, int], Task
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
        artifact_store: Optional[FileSystemArtifactStore] = None,
        current_scope_path: str = "",
        parent_name: str = "",  # Add parent_name parameter
    ) -> List[Dict[str, Any]]:
        if recursion_depth > max_depth:
            logger.warning(
                f"Recursion depth {recursion_depth} exceeds max {max_depth}. Aborting deeper recursion."
            )
            return []

        # Determine if this is a function handler (which doesn't return child items)
        is_function_handler = "function" in str(create_task_fn).lower()

        # Ensure we have a valid artifact store
        if artifact_store is None:
            logger.warning(
                "No artifact store provided to handler, creating default store"
            )
            artifact_store = FileSystemArtifactStore(
                base_path="./artifacts", logger=logger, agent_role="orchestrator"
            )

        results = []
        for name, desc in items[:max_items]:
            if task_counter["count"] >= max_tasks:
                logger.warning(
                    f"Task cap reached ({max_tasks}). Halting further spawning."
                )
                break

            # Compute scope for this item
            item_full_scope_path = os.path.join(
                current_scope_path, name.replace(" ", "_")
            )

            # Determine role (optional: infer from create_task_fn or pass as param)
            role = getattr(create_task_fn, "role", "Agent")

            # Create a scoped child store for this item
            item_store = artifact_store.create_child_store(
                scope_path=item_full_scope_path, agent_role=role
            )

            # Create tools that use the scoped store
            tools = [
                scoped_save_artifact,  # Pass the Tool object directly
                scoped_get_artifact,
            ]

            # Pass tools to agent creation for both main and review tasks
            # If this is a function handler and we know the parent class name, pass it
            task_creator = lambda n, d: create_task_fn(n, d, tools)

            # Special handling for function manager to pass class_name
            if is_function_handler and parent_name:
                task_creator = lambda n, d: create_task_fn(n, d, parent_name, tools)

            # For function outputs, we use a different approach since they don't return JSON lists
            # Function managers are leaf nodes that produce Python code, not structured output
            if is_function_handler:
                # Use run_task_with_review instead of run_task_with_review_and_refine
                # since we don't need to parse or extract items from function outputs
                outputs = run_task_with_review(
                    task_creator(name, desc),
                    lambda t: create_review_fn(t),
                    logger=logger,
                    verbose=verbose,
                )

                if outputs.get("error"):
                    logger.error(
                        f"Critical error in task '{name}': {outputs['error_msg']}"
                    )
                    results.append(
                        {
                            "name": name,
                            "description": desc,
                            "main_output": outputs.get("main", ""),
                            "review": outputs.get("review", ""),
                            "children": [],
                            "error": True,
                            "error_msg": outputs.get("error_msg"),
                        }
                    )
                    continue

                # Save code to output directory
                try:
                    # Save the function's code to the output directory
                    code_output = outputs.get("main", "")
                    if code_output:
                        # Save Python code to output directory
                        output_file_path = os.path.join(
                            current_scope_path, f"{name.replace(' ', '_')}.py"
                        )
                        # Also save a JSON metadata file to the artifact store with a summary
                        function_metadata = {
                            "name": name,
                            "description": desc,
                            "class_name": parent_name,
                            "summary": f"Implementation of {name} function for {parent_name} class",
                        }
                        item_store.save_artifact(
                            f"{name}_metadata.json",
                            json.dumps(function_metadata, indent=2),
                        )

                        # Save the actual code in the artifact store too for reference
                        item_store.save_artifact(f"{name}.py", code_output)

                        # Also save the review as a separate file
                        if outputs.get("review"):
                            item_store.save_artifact(
                                f"{name}_review.txt", outputs.get("review", "")
                            )
                except Exception as e:
                    logger.error(f"Error saving function code: {e}")

                # For function outputs, we don't have children items to process
                results.append(
                    {
                        "name": name,
                        "description": desc,
                        "main_output": outputs.get("main", ""),
                        "review": outputs.get("review", ""),
                        "children": [],
                        "error": False,
                        "error_msg": None,
                    }
                )
            else:
                # Normal processing for non-function outputs (JSON lists)
                outputs = run_task_with_review_and_refine(
                    name,
                    desc,
                    task_creator,
                    lambda t: create_review_fn(t),
                    lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
                        ctf,
                        n,
                        d,
                        oo,
                        ro,
                        cyc,
                        "Output MUST be a JSON array of objects with 'name' and 'description'.",
                    ),
                    parse_fn,
                    logger,
                    verbose,
                    review_cycles,
                    parse_retries,
                    max_items,
                    scoped_store=item_store,  # Pass the scoped store to the task
                )

                if outputs.get("error"):
                    logger.error(
                        f"Critical error in task '{name}': {outputs['error_msg']}"
                    )
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
                    continue

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
                        artifact_store=item_store,  # Pass the item's store to the next level
                        current_scope_path=item_full_scope_path,
                        parent_name=name,  # Pass current name as parent to next level
                    )

                results.append(
                    {
                        "name": name,
                        "description": desc,
                        "cycles": outputs.get("cycles", []),
                        "final_output": outputs.get("final_output"),
                        "final_review": outputs.get("final_review"),
                        "children": next_results,
                        "error": False,
                        "error_msg": None,
                    }
                )

        return results

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

    # Define the create_refine_fn for the architect step
    architect_crf: Callable[
        [Callable[[str, str], Task], str, str, str, str, int], Task
    ] = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf,
        n,
        d,
        oo,
        ro,
        cyc,
        "Output MUST be a JSON array of objects with 'name' and 'description'.",
    )

    # Create tools for the Architect with the root artifact store
    tools = [
        scoped_save_artifact,
        scoped_get_artifact,
    ]

    # Updated to pass tools and artifact_store
    arch_outputs = run_task_with_review_and_refine(
        "System Architecture",
        requirements,
        lambda n, d: create_architect_task(f"{n}: {d}", tools=tools),
        lambda t: create_architect_review_task(t, tools=tools),
        architect_crf,
        extract_items_from_pydantic_output,
        logger,
        verbose,
        review_cycles,
        parse_retries,
        max_items,
        scoped_store=artifact_store,  # Pass the artifact store
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
    # CORRECTED: Use "parsed_list" from arch_outputs
    modules = arch_outputs.get("parsed_list", [])

    # Ensure modules list doesn't exceed max_items, though run_task_with_review_and_refine
    # should have handled this if it used parse_fn.
    if modules:  # Add this check
        modules = modules[:max_items]

    # Build handlers: functions have no children, classes call functions, modules call classes
    function_handler = make_next_level_handler(
        create_function_manager_task,
        create_function_review_task,
        lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
            ctf,
            n,
            d,
            oo,
            ro,
            cyc,
            "Provide the full code implementation with a docstring.",
        ),
        extract_items_from_pydantic_output,
        None,
    )
    class_handler = make_next_level_handler(
        create_class_manager_task,
        create_class_review_task,
        lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
            ctf,
            n,
            d,
            oo,
            ro,
            cyc,
            "Output MUST be a JSON array of objects with 'name' and 'description'.",
        ),
        extract_items_from_pydantic_output,
        function_handler,
    )
    module_handler = make_next_level_handler(
        create_module_manager_task,
        create_module_review_task,
        lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
            ctf,
            n,
            d,
            oo,
            ro,
            cyc,
            "Output MUST be a JSON array of objects with 'name' and 'description'.",
        ),
        extract_items_from_pydantic_output,
        class_handler,
    )

    # 2. Recursive orchestration starting at module level
    module_results = orchestrate_level(
        modules,
        create_module_manager_task,
        create_module_review_task,
        lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
            ctf,
            n,
            d,
            oo,
            ro,
            cyc,
            "Output MUST be a JSON array of objects with 'name' and 'description'.",
        ),
        extract_items_from_pydantic_output,
        module_handler,
        max_depth,
        max_items,
        logger,
        max_tasks,
        verbose,
        review_cycles,
        parse_retries,
        artifact_store,
        "modules",
    )

    # Save individual artifacts (designs, reviews, code, tests)
    try:
        ensure_dir_exists(output_dir)
    except Exception as e:
        logger.error(f"Could not create output directory: {e}")
        return {
            "architecture": architecture_str,
            "architecture_review": architecture_review,
            "modules": module_results,
            "error": True,
            "error_msg": str(e),
        }

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

import logging
from typing import List, Tuple, Dict, Any, Optional, Callable, Union, TypeVar, Type
from pathlib import Path
from crewai import Task, Crew, Process
from convoke.agents import ItemListOutput, check_agent_decision
from convoke.utils import lint_python_code
from convoke.store import FileSystemArtifactStore
from convoke.tools import scoped_get_artifact, scoped_save_artifact
import json
import os
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
from convoke.events import get_event_bus, EventTypes
import copy

__all__ = [
    "parse_json_list",
    "orchestrate_level",
    "orchestrate_full_workflow",
]


task_counter = {"count": 0}


def run_task_with_review(
    task: Dict[str, Any],
    review_task: Dict[str, Any],
    store: FileSystemArtifactStore,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
    previous_task_outputs: Optional[List[Dict[str, Any]]] = None,
    previous_level_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run a task with a single review step.

    Args:
        task: The task definition
        review_task: The review task definition
        store: The artifact store
        output_dir: The output directory
        config: Configuration settings
        logger: Optional logger
        verbose: Verbosity level
        previous_task_outputs: Outputs from previous tasks in the same level
        previous_level_outputs: Outputs from previous levels

    Returns:
        Dictionary containing the task output and review
    """
    # Get event bus
    event_bus = get_event_bus(logger)

    task_name = task.get("name", "Unnamed Task")
    review_name = review_task.get("name", "Review Task")

    if verbose >= 1 and logger:
        logger.info(f"Starting task with review: {task_name}")

    # First run the main task
    task_result = run_task(
        task,
        store,
        output_dir,
        config,
        logger,
        verbose,
        previous_task_outputs,
        previous_level_outputs,
    )

    if task_result["error"]:
        return {
            "error": True,
            "error_msg": task_result["error_msg"],
            "output": None,
            "review": None,
            "output_path": None,
            "review_path": None,
        }

    # Add task output to review task context
    review_task_copy = copy.deepcopy(review_task)
    if "context" not in review_task_copy:
        review_task_copy["context"] = []

    # Add the task output to the review context
    review_task_copy["context"].append(
        {
            "role": "user",
            "content": f"Review the following output:\n\n{task_result['output']}",
        }
    )

    # Execute the review task
    if verbose >= 1 and logger:
        logger.info(f"Starting review task: {review_name}")

    # Emit review task started event
    event_bus.publish(
        EventTypes.TASK_STARTED,
        task_name=review_name,
        task_type="review",
    )

    try:
        # Create the review task object
        review_task_obj = Task(
            description=review_task_copy.get("description", ""),
            expected_output=review_task_copy.get("expected_output", ""),
            agent=review_task_copy.get("agent"),
            context=review_task_copy.get("context", []),
        )

        # Get the agent
        agent = review_task_obj.agent

        if agent is None:
            error_msg = f"Review task {review_name} missing agent"
            if logger:
                logger.error(error_msg)
            return {
                "error": True,
                "error_msg": error_msg,
                "output": task_result["output"],
                "review": None,
                "output_path": task_result["output_path"],
                "review_path": None,
            }

        # Execute the review task
        review_output = agent.execute_task(review_task_obj)
        task_counter["count"] += 1

        # Save the review output if we have a store
        review_path = None
        if store:
            review_path = f"{review_name.replace(' ', '_')}.txt"
            store.save_artifact(review_path, review_output)

        # Emit review task completed event
        event_bus.publish(
            EventTypes.TASK_COMPLETED,
            task_name=review_name,
            task_type="review",
            output_path=review_path,
        )

        return {
            "error": False,
            "error_msg": None,
            "output": task_result["output"],
            "review": review_output,
            "output_path": task_result["output_path"],
            "review_path": review_path,
        }
    except Exception as e:
        error_msg = f"Review task execution failed: {str(e)}"
        if logger:
            logger.error(error_msg)

        # Emit review task error event
        event_bus.publish(
            EventTypes.TASK_ERROR,
            error_msg=error_msg,
            task_name=review_name,
            exception=e,
        )

        return {
            "error": True,
            "error_msg": error_msg,
            "output": task_result["output"],
            "review": None,
            "output_path": task_result["output_path"],
            "review_path": None,
        }


def run_task_with_review_and_refine(
    task: Dict[str, Any],
    store: FileSystemArtifactStore,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
    previous_task_outputs: Optional[List[Dict[str, Any]]] = None,
    previous_level_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run a task with review and refinement cycles."""
    event_bus = get_event_bus(logger)
    task_name = task.get("name", "Unnamed Task")
    task_desc = task.get("description", "")
    review_cycles = task.get("refine", {}).get("cycles", 3)
    parse_retries = task.get("refine", {}).get("parse_retries", 3)
    max_items = task.get("refine", {}).get("max_items", 10)
    parse_fn = task.get("refine", {}).get("parse_fn")
    if parse_fn is None:
        from convoke.parsers import parse_json_list

        parse_fn = parse_json_list

    if verbose >= 1 and logger:
        logger.info(f"Starting task with review and refinement: {task_name}")

    event_bus.publish(
        EventTypes.TASK_STARTED,
        task_name=task_name,
        task_type="review_and_refine",
    )

    try:
        # Create the main task object
        task_obj = Task(
            description=task_desc,
            expected_output=task.get("expected_output", ""),
            agent=task.get("agent"),
            context=task.get("context", []),
        )
        main_agent = task_obj.agent
        if main_agent is None:
            error_msg = f"Task {task_name} missing agent"
            if logger:
                logger.error(error_msg)
            return {
                "error": True,
                "error_msg": error_msg,
                "parsed_list": [],
                "cycles": [],
                "final_output": "",
                "final_review": "",
            }
        if verbose >= 2 and logger:
            logger.info(f"Executing main task: {task_name}")
        main_output = main_agent.execute_task(task_obj)
        task_counter["count"] += 1
        if store:
            store.save_artifact(f"{task_name.replace(' ', '_')}.json", main_output)
        # Create the review task
        review_task_def = task.get("review", {})
        review_task_obj = Task(
            description=review_task_def.get("description", f"Review for {task_name}"),
            expected_output=review_task_def.get("expected_output", ""),
            agent=review_task_def.get("agent"),
            context=review_task_def.get("context", []) + [task_obj],
        )
        review_agent = review_task_obj.agent
        if review_agent is None:
            error_msg = f"Review task for {task_name} missing agent"
            if logger:
                logger.error(error_msg)
            return {
                "error": True,
                "error_msg": error_msg,
                "parsed_list": [],
                "cycles": [],
                "final_output": main_output,
                "final_review": "",
            }
        if verbose >= 2 and logger:
            logger.info(f"Executing review for: {task_name}")
        review_output = review_agent.execute_task(review_task_obj)
        task_counter["count"] += 1
        if store:
            store.save_artifact(
                f"{task_name.replace(' ', '_')}_review.txt", review_output
            )
        # Parse the initial output
        parsed_items = []
        parse_success = False
        parse_attempt = 0
        while parse_attempt < parse_retries and not parse_success:
            parse_attempt += 1
            parsed_items = parse_fn(main_output, max_items)
            parse_success = len(parsed_items) > 0
            if parse_success:
                if verbose >= 2 and logger:
                    logger.info(
                        f"Successfully parsed {len(parsed_items)} items from output"
                    )
                break
            elif parse_attempt < parse_retries:
                if logger:
                    logger.warning(
                        f"Parse attempt {parse_attempt} failed, trying again..."
                    )
        cycles = [
            {
                "cycle": 0,
                "output": main_output,
                "review": review_output,
                "parsed_items_count": len(parsed_items),
            }
        ]
        current_output = main_output
        current_review = review_output
        for cycle in range(1, review_cycles + 1):
            if parse_success or cycle >= review_cycles:
                break
            if verbose >= 1 and logger:
                logger.info(f"Starting refinement cycle {cycle} for {task_name}")
            # Create refinement task
            refine_task_def = task.get("refine", {})
            refine_task_obj = Task(
                description=(
                    f"Refine your previous output for '{task_name}'.\n"
                    f"Original description: {task_desc}\n"
                    f"Your previous output:\n{current_output}\n"
                    f"Peer review feedback:\n{current_review}\n"
                    f"This is refinement cycle {cycle}. Please address the review feedback. "
                    f"Output MUST be a JSON array of objects with 'name' and 'description'."
                ),
                expected_output="A revised output in the specified format.",
                agent=refine_task_def.get("agent", main_agent),
                context=[task_obj],
            )
            refine_agent = refine_task_obj.agent
            if refine_agent is None:
                error_msg = (
                    f"Refinement task for {task_name} cycle {cycle} missing agent"
                )
                if logger:
                    logger.error(error_msg)
                break
            refined_output = refine_agent.execute_task(refine_task_obj)
            task_counter["count"] += 1
            if store:
                store.save_artifact(
                    f"{task_name.replace(' ', '_')}_refined_{cycle}.json",
                    refined_output,
                )
            # Parse the refined output
            parse_success = False
            parse_attempt = 0
            while parse_attempt < parse_retries and not parse_success:
                parse_attempt += 1
                parsed_items = parse_fn(refined_output, max_items)
                parse_success = len(parsed_items) > 0
                if parse_success:
                    if verbose >= 2 and logger:
                        logger.info(
                            f"Successfully parsed {len(parsed_items)} items from refined output"
                        )
                    break
                elif parse_attempt < parse_retries:
                    if logger:
                        logger.warning(
                            f"Parse attempt {parse_attempt} for refinement failed, trying again..."
                        )
            # Get a review of the refined output
            review_task_obj.context = [refine_task_obj]
            refined_review = review_agent.execute_task(review_task_obj)
            task_counter["count"] += 1
            if store:
                store.save_artifact(
                    f"{task_name.replace(' ', '_')}_refined_review_{cycle}.txt",
                    refined_review,
                )
            current_output = refined_output
            current_review = refined_review
            cycles.append(
                {
                    "cycle": cycle,
                    "output": refined_output,
                    "review": refined_review,
                    "parsed_items_count": len(parsed_items),
                }
            )
            if parse_success:
                break
        return {
            "error": False,
            "error_msg": None,
            "parsed_list": parsed_items,
            "cycles": cycles,
            "final_output": current_output,
            "final_review": current_review,
        }
    except Exception as e:
        error_msg = f"Task with review and refinement failed: {str(e)}"
        if logger:
            logger.error(error_msg)
        event_bus.publish(
            EventTypes.TASK_ERROR,
            error_msg=error_msg,
            task_name=task_name,
            exception=e,
        )
        return {
            "error": True,
            "error_msg": error_msg,
            "parsed_list": [],
            "cycles": [],
            "final_output": "",
            "final_review": "",
        }


def create_refinement_task(
    create_task_fn: Callable[[str, str, Optional[List[Any]]], Task],
    name: str,
    desc: str,
    original_output: str,
    review_output: str,
    cycle: int,
    output_format_instruction: str,
    tools: Optional[List[Any]] = None,
) -> Task:
    """
    Create a refinement task that includes previous output, peer feedback, and output format instructions.
    """
    base_task = create_task_fn(name, desc, tools)
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
    create_task_fn: Callable[[str, str, Optional[List[Any]], Optional[str]], Any],
    create_review_fn: Callable[[Any], Any],
    create_refine_fn: Callable[
        [
            Callable[[str, str, Optional[List[Any]], Optional[str]], Any],
            str,
            str,
            str,
            str,
            int,
            str,
            Optional[List[Any]],
        ],
        Any,
    ],
    parse_fn: Callable[[str, int], List[Tuple[str, str]]],
    next_level_fn: Optional[Callable[..., List[Dict[str, Any]]]],
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
            task_creator = lambda n, d, tools=tools: create_task_fn(n, d, tools)

            # Special handling for function manager to pass class_name
            if is_function_handler and parent_name:
                task_creator = lambda n, d, tools=tools: create_task_fn(
                    n, d, tools, parent_name
                )

            # For function outputs, we use a different approach since they don't return JSON lists
            # Function managers are leaf nodes that produce Python code, not structured output
            if is_function_handler:
                # For function handlers, use agent.execute_task directly
                task_obj = task_creator(name, desc)
                review_obj = create_review_fn(task_obj)
                main_output = task_obj.agent.execute_task(task_obj)
                review_output = review_obj.agent.execute_task(review_obj)
                # Save code and review as before
                try:
                    code_output = main_output
                    if code_output:
                        output_file_path = os.path.join(
                            current_scope_path, f"{name.replace(' ', '_')}.py"
                        )
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
                        item_store.save_artifact(f"{name}.py", code_output)
                        if review_output:
                            item_store.save_artifact(
                                f"{name}_review.txt", review_output
                            )
                except Exception as e:
                    logger.error(f"Error saving function code: {e}")
                results.append(
                    {
                        "name": name,
                        "description": desc,
                        "main_output": main_output,
                        "review": review_output,
                        "children": [],
                        "error": False,
                        "error_msg": None,
                    }
                )
            else:
                # Normal processing for non-function outputs (JSON lists)
                task_obj = task_creator(name, desc)
                outputs = run_task_with_review_and_refine(
                    task_obj,
                    item_store,
                    Path(current_scope_path),
                    {},
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
    next_level_fn: Optional[Callable[..., List[Dict[str, Any]]]],
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


def orchestrate_level(
    level_def: Dict[str, Any],
    store: FileSystemArtifactStore,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
    previous_level_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Orchestrate a single level of a workflow, including its tasks and reviews."""
    # Get event bus
    event_bus = get_event_bus(logger)

    level_name = level_def.get("name", "Unnamed Level")

    if verbose >= 1 and logger:
        logger.info(f"Starting level orchestration: {level_name}")

    task_outputs: List[Dict[str, Any]] = []
    previous_level_outputs = previous_level_outputs or []

    # Prepare tasks
    tasks = level_def.get("tasks", [])

    if not tasks:
        error_msg = f"Level {level_name} has no tasks defined."
        if logger:
            logger.error(error_msg)

        # Emit task error event
        event_bus.publish(
            EventTypes.LEVEL_ERROR,
            error_msg=error_msg,
            level_name=level_name,
        )

        return {
            "error": True,
            "error_msg": error_msg,
            "tasks": [],
        }

    # Process each task in sequence
    for task_num, task in enumerate(tasks, 1):
        task_name = task.get("name", f"Task {task_num}")

        if verbose >= 1 and logger:
            logger.info(f"Starting task {task_num}: {task_name}")

        # Emit task started event
        event_bus.publish(
            EventTypes.TASK_STARTED,
            task_name=task_name,
            task_num=task_num,
            level_name=level_name,
        )

        try:
            # Determine task execution function based on review and refine settings
            if task.get("review", {}).get("enabled", False) and task.get(
                "refine", {}
            ).get("enabled", False):
                task_output = run_task_with_review_and_refine(
                    task,
                    store,
                    output_dir,
                    config=config,
                    logger=logger,
                    verbose=verbose,
                    previous_task_outputs=task_outputs,
                    previous_level_outputs=previous_level_outputs,
                )
            elif task.get("review", {}).get("enabled", False):
                review_task = task.get("review", {})
                task_output = run_task_with_review(
                    task,
                    review_task,
                    store,
                    output_dir,
                    config,
                    logger,
                    verbose,
                    previous_task_outputs=task_outputs,
                    previous_level_outputs=previous_level_outputs,
                )
            else:
                task_output = run_task(
                    task,
                    store,
                    output_dir,
                    config,
                    logger,
                    verbose,
                    previous_task_outputs=task_outputs,
                    previous_level_outputs=previous_level_outputs,
                )

            task_outputs.append(task_output)

            # Emit task completed event
            event_bus.publish(
                EventTypes.TASK_COMPLETED,
                task_name=task_name,
                task_num=task_num,
                level_name=level_name,
                output_path=task_output.get("output_path"),
            )

        except Exception as e:
            error_msg = f"Task {task_name} failed: {str(e)}"
            if logger:
                logger.error(error_msg)

            # Emit task error event
            event_bus.publish(
                EventTypes.TASK_ERROR,
                error_msg=error_msg,
                task_name=task_name,
                task_num=task_num,
                level_name=level_name,
                exception=e,
            )

            task_outputs.append(
                {
                    "error": True,
                    "error_msg": error_msg,
                    "output": None,
                    "output_path": None,
                }
            )

            # Determine if we should continue or stop based on config
            should_continue = config.get("continue_on_task_error", False)
            if not should_continue:
                # Emit level error event
                event_bus.publish(
                    EventTypes.LEVEL_ERROR,
                    error_msg=f"Level stopped due to task error: {error_msg}",
                    level_name=level_name,
                )

                return {
                    "error": True,
                    "error_msg": f"Level stopped due to task error: {error_msg}",
                    "tasks": task_outputs,
                }

    return {
        "error": False,
        "error_msg": None,
        "tasks": task_outputs,
    }


def orchestrate_full_workflow(
    workflow_def: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """Orchestrate a full workflow, including multiple levels with their tasks and reviews."""
    # Get event bus
    event_bus = get_event_bus(logger)

    if "name" in workflow_def:
        workflow_name = workflow_def["name"]
    else:
        workflow_name = "Full Workflow"

    # Emit workflow started event
    event_bus.publish(
        EventTypes.WORKFLOW_STARTED,
        workflow_name=workflow_name,
        level_count=len(workflow_def.get("levels", [])),
    )

    try:
        # Create store for artifacts
        store_config = config.get("store", {})
        store_instance = FileSystemArtifactStore(**store_config)

        # Prepare workflow levels
        levels = workflow_def.get("levels", [])

        if not levels:
            error_msg = "Workflow has no levels defined."
            if logger:
                logger.error(error_msg)

            # Emit error event
            event_bus.publish(
                EventTypes.WORKFLOW_ERROR,
                error_msg=error_msg,
                workflow_name=workflow_name,
            )

            return {
                "error": True,
                "error_msg": error_msg,
                "levels": [],
            }

        # Process each level in sequence
        level_outputs: List[Dict[str, Any]] = []
        for level_num, level in enumerate(levels, 1):
            if verbose >= 1 and logger:
                logger.info(
                    f"Starting level {level_num}: {level.get('name', 'Unnamed Level')}"
                )

            # Emit level started event
            event_bus.publish(
                EventTypes.LEVEL_STARTED,
                level_name=level.get("name", f"Level {level_num}"),
                level_num=level_num,
                workflow_name=workflow_name,
            )

            try:
                level_output = orchestrate_level(
                    level,
                    store_instance,
                    output_dir,
                    config=config,
                    logger=logger,
                    verbose=verbose,
                    previous_level_outputs=level_outputs if level_num > 1 else [],
                )
                level_outputs.append(level_output)

                # Emit level completed event
                event_bus.publish(
                    EventTypes.LEVEL_COMPLETED,
                    level_name=level.get("name", f"Level {level_num}"),
                    level_num=level_num,
                    workflow_name=workflow_name,
                    task_count=len(level_output.get("tasks", [])),
                )

            except Exception as e:
                error_msg = f"Level {level_num} failed: {str(e)}"
                if logger:
                    logger.error(error_msg)

                # Emit error event
                event_bus.publish(
                    EventTypes.LEVEL_ERROR,
                    error_msg=error_msg,
                    level_name=level.get("name", f"Level {level_num}"),
                    level_num=level_num,
                    workflow_name=workflow_name,
                    exception=e,
                )

                level_outputs.append(
                    {
                        "error": True,
                        "error_msg": error_msg,
                        "tasks": [],
                    }
                )

                # Determine if we should continue or stop based on config
                should_continue = config.get("continue_on_level_error", False)
                if not should_continue:
                    # Emit workflow error event
                    event_bus.publish(
                        EventTypes.WORKFLOW_ERROR,
                        error_msg=f"Workflow stopped due to level error: {error_msg}",
                        workflow_name=workflow_name,
                        level_num=level_num,
                    )

                    return {
                        "error": True,
                        "error_msg": f"Workflow stopped due to level error: {error_msg}",
                        "levels": level_outputs,
                    }

        # Emit workflow completed event
        event_bus.publish(
            EventTypes.WORKFLOW_COMPLETED,
            workflow_name=workflow_name,
            level_count=len(level_outputs),
            success_count=sum(
                1 for level in level_outputs if not level.get("error", False)
            ),
            error_count=sum(1 for level in level_outputs if level.get("error", False)),
        )

        # === Artifact saving (project scaffolding/tree) with error handling ===
        try:
            # Attempt to extract architecture and modules from outputs
            architecture = None
            modules = None
            for level in level_outputs:
                if not architecture and level.get("tasks"):
                    for task in level["tasks"]:
                        if task.get("final_output"):
                            architecture = task["final_output"]
                            break
                if not modules and level.get("tasks"):
                    for task in level["tasks"]:
                        if task.get("parsed_list"):
                            # Convert parsed_list to module dicts
                            modules = [
                                {"name": n, "description": d}
                                for n, d in task["parsed_list"]
                            ]
                            break
            if architecture is None:
                architecture = ""
            if modules is None:
                modules = []
            write_project_scaffolding(str(output_dir), architecture, modules)
            write_tree_visualization(str(output_dir), modules)
        except Exception as e:
            error_msg = f"Artifact saving failed: {e}"
            if logger:
                logger.error(error_msg)
            return {
                "error": True,
                "error_msg": error_msg,
                "levels": level_outputs,
            }

        return {
            "error": False,
            "error_msg": None,
            "levels": level_outputs,
        }

    except Exception as e:
        error_msg = f"Workflow orchestration failed: {str(e)}"
        if logger:
            logger.error(error_msg)

        # Emit workflow error event
        event_bus.publish(
            EventTypes.WORKFLOW_ERROR,
            error_msg=error_msg,
            workflow_name=workflow_name,
            exception=e,
        )

        return {
            "error": True,
            "error_msg": error_msg,
            "levels": [],
        }


def run_task_with_review_and_dynamic_refine(
    main_task: Task,
    review_task_fn: Callable[[Task], Task],
    refine_task_fn: Callable[[Task, Task], Task],
    max_iterations: int = 5,
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
) -> Dict[str, Any]:
    """
    Run a task with review and refinement, where the agent can decide whether to continue iterating.
    The agent can indicate its decision using: @agent Continue: "Yes/No question or statement"

    Args:
        main_task: The main task to execute
        review_task_fn: Function to create a review task from the main task
        refine_task_fn: Function to create a refinement task from the main and review tasks
        max_iterations: Maximum number of iterations to perform
        logger: Optional logger
        verbose: Verbosity level

    Returns:
        Dictionary containing outputs from the tasks and iteration information
    """
    try:
        # Get event bus
        event_bus = get_event_bus(logger)

        # Emit workflow started event
        event_bus.publish(
            EventTypes.WORKFLOW_STARTED,
            workflow_name=f"Task-Review-DynamicRefine Workflow for {getattr(main_task, 'name', 'Main Task')}",
            max_iterations=max_iterations,
        )

        main_agent = main_task.agent
        if main_agent is None:
            error_msg = "Main task is missing an agent."
            if logger:
                logger.error(error_msg)

            # Emit error event
            event_bus.publish(
                EventTypes.WORKFLOW_ERROR,
                error_msg=error_msg,
                workflow_name=f"Task-Review-DynamicRefine Workflow for {getattr(main_task, 'name', 'Main Task')}",
            )

            return {
                "main": "",
                "review": f"ERROR: {error_msg}",
                "iterations": 0,
                "error": True,
                "error_msg": error_msg,
            }

        # Run initial task
        # Emit task started event
        event_bus.publish(
            EventTypes.TASK_STARTED,
            task_name=getattr(main_task, "name", "Main Task"),
            agent_name=getattr(main_agent, "name", "Main Agent"),
        )

        main_output = main_agent.execute_task(main_task)
        task_counter["count"] += 1

        # Emit task completed event
        event_bus.publish(
            EventTypes.TASK_COMPLETED,
            task_name=getattr(main_task, "name", "Main Task"),
            task_type="main",
            output_summary=(
                main_output[:100] + "..." if len(main_output) > 100 else main_output
            ),
        )

        review_task = review_task_fn(main_task)
        review_agent = review_task.agent

        if review_agent is None:
            error_msg = "Review task is missing an agent."
            if logger:
                logger.error(error_msg)

            # Emit error event
            event_bus.publish(
                EventTypes.WORKFLOW_ERROR,
                error_msg=error_msg,
                workflow_name=f"Task-Review-DynamicRefine Workflow for {getattr(main_task, 'name', 'Main Task')}",
            )

            return {
                "main": main_output,
                "review": f"ERROR: {error_msg}",
                "iterations": 0,
                "error": True,
                "error_msg": error_msg,
            }

        # Emit task started event
        event_bus.publish(
            EventTypes.TASK_STARTED,
            task_name=getattr(review_task, "name", "Review Task"),
            agent_name=getattr(review_agent, "name", "Review Agent"),
        )

        review_output = review_agent.execute_task(review_task)
        task_counter["count"] += 1

        # Emit task completed event
        event_bus.publish(
            EventTypes.TASK_COMPLETED,
            task_name=getattr(review_task, "name", "Review Task"),
            task_type="review",
            output_summary=(
                review_output[:100] + "..."
                if len(review_output) > 100
                else review_output
            ),
        )

        iterations = 0
        refined_outputs = []

        # Import the agent decision checker
        from convoke.agents import check_agent_decision

        while iterations < max_iterations:
            event_bus.publish(
                EventTypes.REFINEMENT_ITERATION_STARTED,
                iteration=iterations + 1,
                max_iterations=max_iterations,
                task_name=getattr(main_task, "name", "Main Task"),
            )

            refine_task = refine_task_fn(main_task, review_task)
            refine_agent = refine_task.agent

            if refine_agent is None:
                error_msg = "Refine task is missing an agent."
                if logger:
                    logger.error(error_msg)

                # Emit error event
                event_bus.publish(
                    EventTypes.WORKFLOW_ERROR,
                    error_msg=error_msg,
                    workflow_name=f"Task-Review-DynamicRefine Workflow for {getattr(main_task, 'name', 'Main Task')}",
                    iteration=iterations + 1,
                )

                break

            # Emit task started event
            event_bus.publish(
                EventTypes.TASK_STARTED,
                task_name=getattr(refine_task, "name", "Refine Task"),
                agent_name=getattr(refine_agent, "name", "Refine Agent"),
                iteration=iterations + 1,
            )

            refined_output = refine_agent.execute_task(refine_task)
            refined_outputs.append(refined_output)
            task_counter["count"] += 1

            # Emit task completed event
            event_bus.publish(
                EventTypes.TASK_COMPLETED,
                task_name=getattr(refine_task, "name", "Refine Task"),
                task_type="refine",
                output_summary=(
                    refined_output[:100] + "..."
                    if len(refined_output) > 100
                    else refined_output
                ),
                iteration=iterations + 1,
            )

            event_bus.publish(
                EventTypes.REFINEMENT_ITERATION_COMPLETED,
                iteration=iterations + 1,
                max_iterations=max_iterations,
                task_name=getattr(main_task, "name", "Main Task"),
            )

            iterations += 1

            # Update the main task for the next iteration
            if not isinstance(main_task.context, list):
                main_task.context = []
            main_task.context.append(refine_task)

            # Check if the agent wants to continue based on its output
            should_continue = check_agent_decision(
                refined_output, default_value=iterations < max_iterations
            )

            if not should_continue:
                if logger and verbose >= 1:
                    logger.info(
                        f"Agent decided to stop refinement after {iterations} iterations"
                    )

                event_bus.publish(
                    EventTypes.AGENT_DECISION,
                    decision="stop",
                    iteration=iterations,
                    task_name=getattr(main_task, "name", "Main Task"),
                )
                break

            # If we're at max iterations, break
            if iterations >= max_iterations:
                break

            # Get a new review
            # Emit task started event
            event_bus.publish(
                EventTypes.TASK_STARTED,
                task_name=getattr(review_task, "name", "Review Task"),
                agent_name=getattr(review_agent, "name", "Review Agent"),
                iteration=iterations,
            )

            review_output = review_agent.execute_task(review_task)
            task_counter["count"] += 1

            # Emit task completed event
            event_bus.publish(
                EventTypes.TASK_COMPLETED,
                task_name=getattr(review_task, "name", "Review Task"),
                task_type="review",
                output_summary=(
                    review_output[:100] + "..."
                    if len(review_output) > 100
                    else review_output
                ),
                iteration=iterations,
            )

        # Emit workflow completed event
        event_bus.publish(
            EventTypes.WORKFLOW_COMPLETED,
            workflow_name=f"Task-Review-DynamicRefine Workflow for {getattr(main_task, 'name', 'Main Task')}",
            iterations_completed=iterations,
            max_iterations=max_iterations,
        )

        return {
            "main": main_output,
            "review": review_output,
            "refined": refined_outputs[-1] if refined_outputs else "",
            "all_refined": refined_outputs,
            "iterations": iterations,
            "error": False,
            "error_msg": None,
        }
    except Exception as e:
        if logger:
            logger.error(f"Task, review, or refine failed: {e}")

        # Emit error event
        event_bus.publish(
            EventTypes.WORKFLOW_ERROR,
            error_msg=str(e),
            workflow_name=f"Task-Review-DynamicRefine Workflow for {getattr(main_task, 'name', 'Main Task')}",
            exception=e,
        )

        return {
            "main": "",
            "review": "",
            "refined": "",
            "error": True,
            "error_msg": str(e),
            "iterations": 0,
        }


def run_task(
    task: Dict[str, Any],
    store: FileSystemArtifactStore,
    output_dir: Path,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
    verbose: int = 0,
    previous_task_outputs: Optional[List[Dict[str, Any]]] = None,
    previous_level_outputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run a single task and return its output.

    Args:
        task: The task definition
        store: The artifact store
        output_dir: The output directory
        config: Configuration settings
        logger: Optional logger
        verbose: Verbosity level
        previous_task_outputs: Outputs from previous tasks in the same level
        previous_level_outputs: Outputs from previous levels

    Returns:
        Dictionary containing the task output
    """
    # Get event bus
    event_bus = get_event_bus(logger)

    task_name = task.get("name", "Unnamed Task")

    if verbose >= 1 and logger:
        logger.info(f"Starting task: {task_name}")

    # Emit task started event
    event_bus.publish(
        EventTypes.TASK_STARTED,
        task_name=task_name,
        task_type="task",
    )

    try:
        # Create the task object
        task_obj = Task(
            description=task.get("description", ""),
            expected_output=task.get("expected_output", ""),
            agent=task.get("agent"),
            context=task.get("context", []),
        )

        # Get the agent
        agent = task_obj.agent

        if agent is None:
            error_msg = f"Task {task_name} missing agent"
            if logger:
                logger.error(error_msg)
            return {
                "error": True,
                "error_msg": error_msg,
                "output": None,
                "output_path": None,
            }

        # Execute the task
        output = agent.execute_task(task_obj)
        task_counter["count"] += 1

        # Save the output if we have a store
        output_path = None
        if store:
            output_path = f"{task_name.replace(' ', '_')}.txt"
            store.save_artifact(output_path, output)

        # Emit task completed event
        event_bus.publish(
            EventTypes.TASK_COMPLETED,
            task_name=task_name,
            task_type="task",
            output_path=output_path,
        )

        return {
            "error": False,
            "error_msg": None,
            "output": output,
            "output_path": output_path,
        }
    except Exception as e:
        error_msg = f"Task execution failed: {str(e)}"
        if logger:
            logger.error(error_msg)

        # Emit task error event
        event_bus.publish(
            EventTypes.TASK_ERROR,
            error_msg=error_msg,
            task_name=task_name,
            exception=e,
        )

        return {
            "error": True,
            "error_msg": error_msg,
            "output": None,
            "output_path": None,
        }

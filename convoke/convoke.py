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

load_dotenv()


class ItemDetail(BaseModel):
    name: str = Field(description="Name of the item (e.g., module, class, function)")
    description: str = Field(description="Purpose or description of the item")


class ItemListOutput(BaseModel):
    items: List[ItemDetail]


# --- Agent Definitions ---


def create_architect_agent():
    """Create the Systems Architect agent."""
    return Agent(
        role="Systems Architect",
        goal="Design a robust, modular software system for the given requirements.",
        backstory=(
            "You are a highly experienced systems architect. You break down complex requirements "
            "into high-level modules, ensuring scalability and maintainability."
        ),
        verbose=True,
        allow_delegation=False,  # Temporarily disable delegation/tools for debugging
        model="gpt-4o",  # Specialised LLM for architect
        # tools=[]
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
        model="gpt-3.5-turbo",  # Specialised LLM for function manager
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


def create_test_developer_agent():
    return Agent(
        role="Test Developer",
        goal="Write a unit test for the given function implementation.",
        backstory="You are responsible for writing clear, effective unit tests for Python functions, using pytest conventions.",
        verbose=True,
        allow_delegation=False,
        model="gpt-3.5-turbo",  # Specialised LLM for test developer
    )


def create_test_reviewer_agent():
    return Agent(
        role="Test Reviewer",
        goal="Critique and enhance the unit test for correctness and coverage.",
        backstory="You are a peer test developer, skilled at reviewing unit tests for completeness and effectiveness.",
        verbose=True,
        allow_delegation=False,
    )


# --- Task Definitions ---


def create_architect_task(requirements):
    """Create the Systems Architect's task."""
    return Task(
        description=(
            f"Analyze the following requirements and design a modular system. "
            f"Your ENTIRE response must be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields. "
            f"Example: {{'items': [{{'name': 'ExampleModule', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself.\n"
            f"Requirements: {requirements}"
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name' and 'description' fields, strictly conforming to the ItemListOutput schema. No extra text.",
        agent=create_architect_agent(),
        output_pydantic=ItemListOutput,
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
        human_input=True,  # Require human approval for architect review
    )


def create_module_manager_task(module_name, module_description):
    """Create a Module Manager's task."""
    return Task(
        description=(
            f"Design the module '{module_name}': {module_description}. "
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields. "
            f"Example: {{'items': [{{'name': 'ExampleClass', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself."
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name' and 'description' fields, strictly conforming to the ItemListOutput schema. No extra text.",
        agent=create_module_manager_agent(),
        output_pydantic=ItemListOutput,
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
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields. "
            f"Example: {{'items': [{{'name': 'ExampleFunction', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself."
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name' and 'description' fields, strictly conforming to the ItemListOutput schema. No extra text.",
        agent=create_class_manager_agent(),
        output_pydantic=ItemListOutput,
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


def create_test_developer_task(function_name, function_code):
    return Task(
        description=(
            f"Write a pytest-style unit test for the following function '{function_name}'. "
            f"The test should be in a single code block, with no extra text.\n"
            f"Function implementation:\n{function_code}"
        ),
        expected_output="A complete pytest-style unit test for the function, as a code block.",
        agent=create_test_developer_agent(),
    )


def create_test_review_task(test_task):
    return Task(
        description="Review the unit test for correctness, coverage, and clarity. Suggest improvements if needed.",
        expected_output="A critique and enhancement suggestions for the unit test.",
        agent=create_test_reviewer_agent(),
        context=[test_task],
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


def orchestrate_full_workflow(
    requirements: str,
    max_depth: int = 4,
    max_items: int = 5,
    max_tasks: int = 100,
    verbose: int = 0,
    logger: Optional[logging.Logger] = None,
    review_cycles: int = 1,
    parse_retries: int = 1,
    output_dir: str = "output",
    artifact_store: Optional[FileSystemArtifactStore] = None,
    get_tool: Optional[ScopedGetArtifactTool] = None,
    save_tool: Optional[ScopedSaveArtifactTool] = None,
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
        output_dir (str): Directory to save the outputs.
        artifact_store (FileSystemArtifactStore, optional): Artifact store instance.
    Returns:
        Dict[str, Any]: Nested results for all levels.
    """
    # Use provided scoped tools or fall back to direct store
    reader = get_tool or ScopedGetArtifactTool(
        store=artifact_store, agent_role="System", allowed_read_prefixes=[""]
    )
    writer = save_tool or ScopedSaveArtifactTool(
        store=artifact_store, agent_role="System", allowed_write_prefixes=[""]
    )

    logger = logger or logging.getLogger(__name__)
    task_counter["count"] = 0
    # Architect-level refinement
    json_output_instruction = (
        "Output MUST be a JSON array of objects, each with 'name' and 'description' fields, "
        "e.g. [{'name': 'ModuleName', 'description': 'Purpose of the module'}, ...] (no extra text, no explanation)."
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
    if not modules and arch_outputs["final_output"]:
        logger.error(
            "Architecture parsing resulted in no modules. Halting module processing."
        )
        return {
            "architecture": arch_outputs["final_output"],
            "architecture_review": arch_outputs["final_review"],
            "architecture_cycles": arch_outputs["cycles"],
            "modules": [],
            "error": True,
            "error_msg": "Failed to parse modules from architect's JSON output.",
        }
    # Ensure parsed architecture JSON dict for saving
    try:
        arch_dict = json.loads(arch_outputs["final_output"])
    except Exception:
        arch_dict = {}
        logger.warning(
            "Failed to parse architecture JSON for arch_dict; using empty dict."
        )
    numbered_list_instruction = "Output in the required format: Each item MUST be in the format '1. Name: Description' (one per line, no extra text)."
    default_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, numbered_list_instruction
    )
    json_aware_create_refine_fn = architect_create_refine_fn
    function_code_instruction = "Provide the full code implementation with a docstring, ensuring it meets the original requirements and addresses the review feedback."
    function_create_refine_fn = lambda ctf, n, d, oo, ro, cyc: create_refinement_task(
        ctf, n, d, oo, ro, cyc, function_code_instruction
    )
    next_function_level = make_next_level_handler(
        create_function_manager_task,
        create_function_review_task,
        function_create_refine_fn,
        lambda x, _: [],
        None,
    )
    next_class_level = make_next_level_handler(
        create_class_manager_task,
        create_class_review_task,
        json_aware_create_refine_fn,
        lambda text, max_items_arg: parse_json_list(text, max_items_arg, logger=logger),
        next_function_level,
    )
    next_module_level = make_next_level_handler(
        create_module_manager_task,
        create_module_review_task,
        json_aware_create_refine_fn,
        lambda text, max_items_arg: parse_json_list(text, max_items_arg, logger=logger),
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
    # Instantiate store
    store = artifact_store  # passed in from main, or create new if None

    # Save architecture via scoped writer
    writer._run("architecture.json", json.dumps(arch_dict, indent=2), True)
    writer._run("architecture_review.txt", arch_outputs["final_review"] or "", False)

    def save_module_tree(modules, parent_dir):
        for mod in modules:
            mod_rel = f"modules/{mod['name'].replace(' ', '_')}/"
            # Save module design and review
            writer._run(
                mod_rel + "module_design.json",
                json.dumps(
                    {
                        "name": mod["name"],
                        "description": mod["description"],
                        "cycles": mod["cycles"],
                        "final_output": mod["final_output"],
                    },
                    indent=2,
                ),
                True,
            )
            writer._run(mod_rel + "module_review.txt", mod["final_review"] or "", False)
            for cls in mod.get("children", []):
                cls_rel = mod_rel + f"{cls['name'].replace(' ', '_')}/"
                # Save class design and review
                writer._run(
                    cls_rel + "class_design.json",
                    json.dumps(
                        {
                            "name": cls["name"],
                            "description": cls["description"],
                            "cycles": cls["cycles"],
                            "final_output": cls["final_output"],
                        },
                        indent=2,
                    ),
                    True,
                )
                writer._run(
                    cls_rel + "class_review.txt", cls["final_review"] or "", False
                )
                for fn in cls.get("children", []):
                    # Save function code and review
                    fn_file = cls_rel + f"{fn['name'].replace(' ', '_')}.py"
                    writer._run(fn_file, fn["final_output"] or "", False)
                    writer._run(
                        cls_rel + f"{fn['name'].replace(' ', '_')}_review.txt",
                        fn["final_review"] or "",
                        False,
                    )
                    # Lint the function code and save lint results
                    if fn["final_output"]:
                        lint_result = lint_python_code(fn["final_output"])
                        writer._run(
                            cls_rel + f"{fn['name'].replace(' ', '_')}_lint.txt",
                            lint_result,
                            False,
                        )
                        # Generate and review unit test
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
                        # Save test code and review
                        test_file = cls_rel + f"test_{fn['name'].replace(' ', '_')}.py"
                        writer._run(test_file, test_code, False)
                        writer._run(
                            cls_rel + f"test_{fn['name'].replace(' ', '_')}_review.txt",
                            test_review,
                            False,
                        )

    save_module_tree(modules_result, output_dir)
    # Project scaffolding
    write_project_scaffolding(output_dir, arch_outputs["final_output"], modules_result)
    # Visualization
    write_tree_visualization(output_dir, modules_result)
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


def ensure_dir_exists(path: str):
    os.makedirs(path, exist_ok=True)


def write_json_file(path: str, data: dict):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_text_file(path: str, text: str):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


import subprocess


def lint_python_code(code: str) -> str:
    """Run flake8 linter on the given Python code string and return the linting output."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp.flush()
        tmp_path = tmp.name
    try:
        result = subprocess.run(["flake8", tmp_path], capture_output=True, text=True)
        lint_output = result.stdout + result.stderr
    finally:
        os.unlink(tmp_path)
    return lint_output.strip()


def write_project_scaffolding(
    output_dir: str, architecture: str, modules: list
) -> None:
    # README.md
    readme_path = os.path.join(output_dir, "README.md")
    module_list = "\n".join(f"- {mod['name']}: {mod['description']}" for mod in modules)
    readme_content = f"""# Generated Project\n\n## System Architecture\n\n{architecture}\n\n## Modules\n\n{module_list}\n\n## How to Run Tests\n\n```bash\npytest\n```\n"""
    write_text_file(readme_path, readme_content)
    # requirements.txt
    requirements_path = os.path.join(output_dir, "requirements.txt")
    requirements_content = "pytest\nflake8\n"
    write_text_file(requirements_path, requirements_content)
    # .gitignore
    gitignore_path = os.path.join(output_dir, ".gitignore")
    gitignore_content = "__pycache__/\n*.pyc\n.env\noutput/\n"
    write_text_file(gitignore_path, gitignore_content)


def write_tree_visualization(output_dir: str, modules: list):
    """Write a text-based tree of the generated project structure."""

    def walk_module_tree(modules, prefix=""):
        lines = []
        for mod in modules:
            mod_name = mod["name"].replace(" ", "_")
            lines.append(f"{prefix}├─ {mod_name}/")
            for cls in mod.get("children", []):
                cls_name = cls["name"].replace(" ", "_")
                lines.append(f"{prefix}│   ├─ {cls_name}/")
                for fn in cls.get("children", []):
                    fn_name = fn["name"].replace(" ", "_")
                    lines.append(f"{prefix}│   │   ├─ {fn_name}.py")
                    lines.append(f"{prefix}│   │   ├─ test_{fn_name}.py")
        return lines

    tree_lines = ["output/"] + walk_module_tree(modules)
    tree_path = os.path.join(output_dir, "PROJECT_TREE.txt")
    write_text_file(tree_path, "\n".join(tree_lines))


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

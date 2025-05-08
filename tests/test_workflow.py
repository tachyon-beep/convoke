import tempfile
import os
import logging
import json
import pytest
from convoke.workflow import parse_numbered_list, parse_json_list, orchestrate_level
from convoke.workflow import orchestrate_full_workflow, run_task_with_review

import convoke.workflow

convoke.workflow.Crew.kickoff = lambda self: None

# Patch run_task_with_review to add debug print
orig_run_task_with_review = run_task_with_review


def debug_run_task_with_review(main_task, review_task_fn, logger=None, verbose=0):
    print(
        f"DEBUG: main_task.output.raw={getattr(main_task.output, 'raw', None)} raw_output={getattr(main_task.output, 'raw_output', None)}"
    )
    return {
        "main": getattr(main_task.output, "raw_output", ""),
        "review": "reviewed",
        "error": False,
        "error_msg": None,
    }


convoke.workflow.run_task_with_review = debug_run_task_with_review


def test_parse_numbered_list_basic():
    text = """
    1. Alpha: First item
    2. Beta: Second item
    3. Gamma: Third item
    """
    items = parse_numbered_list(text)
    assert items == [
        ("Alpha", "First item"),
        ("Beta", "Second item"),
        ("Gamma", "Third item"),
    ]


def test_parse_numbered_list_multiline():
    text = """
    1. Alpha: First item
       More details here.
    2. Beta: Second item
    """
    items = parse_numbered_list(text)
    assert items[0][0] == "Alpha"
    assert "More details" in items[0][1]


def test_parse_json_list_valid():
    text = '[{"name": "A", "description": "desc A"}, {"name": "B", "description": "desc B"}]'
    items = parse_json_list(text)
    assert items == [("A", "desc A"), ("B", "desc B")]


def test_parse_json_list_invalid():
    text = "not a json list"
    items = parse_json_list(text)
    assert items == []


def test_orchestrate_level_minimal():
    # Dummy task functions for dry-run
    def dummy_task_fn(name, desc):
        class DummyTask:
            agent = object()
            output = type(
                "Out",
                (),
                {
                    "pydantic": None,
                    "raw": json.dumps(
                        [{"name": name + "X", "description": desc + "X"}]
                    ),
                },
            )()

        return DummyTask()

    def dummy_review_fn(task):
        class DummyReview:
            agent = object()
            output = type("Out", (), {"raw_output": "reviewed"})()

        return DummyReview()

    def dummy_refine_fn(*a, **k):
        return dummy_task_fn(a[1], a[2])

    def dummy_parse_fn(text, max_items):
        return [("Child", "desc")] if text else []

    logger = logging.getLogger("test")
    # Call orchestrate_level using positional arguments to match current signature
    results = orchestrate_level(
        [("Parent", "desc")],
        dummy_task_fn,
        dummy_review_fn,
        dummy_refine_fn,
        dummy_parse_fn,
        1,  # recursion_depth
        2,  # max_depth
        2,  # max_items
        logger,
        10,  # max_tasks
        0,  # verbose
        1,  # review_cycles
        1,  # parse_retries
    )
    assert results[0]["name"] == "Parent"
    assert results[0]["children"] == []


class DummyArtifactStore:
    def __init__(self, *a, **k):
        pass


def test_orchestrate_full_workflow_minimal(monkeypatch, tmp_path):
    # Dummy outputs for each level
    arch_json = '[{"name": "ModuleA", "description": "DescA"}]'
    mod_json = '[{"name": "ClassA", "description": "DescClassA"}]'
    cls_json = '[{"name": "funcA", "description": "DescFuncA"}]'
    fn_code = "def funcA():\n    pass"

    # Dummy task classes
    class DummyTask:
        def __init__(self, raw):
            print(f"DEBUG: DummyTask.__init__ raw={raw}")
            self.agent = object()
            self.output = type(
                "Out", (), {"pydantic": None, "raw": raw, "raw_output": raw}
            )()

    class DummyReview:
        agent = object()
        output = type("Out", (), {"raw_output": "reviewed"})()

    # Patch agent/task creation and run_task_with_review in convoke.workflow
    monkeypatch.setattr(
        "convoke.workflow.create_architect_task", lambda req: DummyTask(arch_json)
    )
    monkeypatch.setattr(
        "convoke.workflow.create_architect_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_module_manager_task", lambda n, d: DummyTask(mod_json)
    )
    monkeypatch.setattr(
        "convoke.workflow.create_module_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_class_manager_task", lambda n, d: DummyTask(cls_json)
    )
    monkeypatch.setattr(
        "convoke.workflow.create_class_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_function_manager_task", lambda n, d: DummyTask(fn_code)
    )
    monkeypatch.setattr(
        "convoke.workflow.create_function_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.write_project_scaffolding", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "convoke.workflow.write_tree_visualization", lambda *a, **k: None
    )
    # Patch run_task_with_review_and_refine to return appropriate outputs per task name
    mapping = {
        "System Architecture": arch_json,
        "ModuleA": mod_json,
        "ClassA": cls_json,
        "funcA": fn_code,
    }
    # Also create mapping for parsed lists
    parsed_mapping = {
        "System Architecture": [("ModuleA", "DescA")],
        "ModuleA": [("ClassA", "DescClassA")],
        "ClassA": [("funcA", "DescFuncA")],
        "funcA": [],
    }
    monkeypatch.setattr(
        "convoke.workflow.run_task_with_review_and_refine",
        lambda name, desc, *args, **kwargs: {
            "final_output": mapping.get(name, ""),
            "final_review": "reviewed",
            "parsed_list": parsed_mapping.get(name, []),
            "error": False,
            "error_msg": None,
        },
    )
    logger = logging.getLogger("test")
    result = orchestrate_full_workflow(
        requirements="reqs",
        max_depth=2,
        max_items=1,
        max_tasks=10,
        verbose=0,
        logger=logger,
        review_cycles=1,
        parse_retries=1,
        output_dir=str(tmp_path),
        artifact_store=object(),
        get_tool=None,
        save_tool=None,
    )
    # Assert the structure is as expected
    assert result["architecture"] == arch_json
    assert result["architecture_review"] == "reviewed"
    assert len(result["modules"]) == 1
    mod = result["modules"][0]
    assert mod["name"] == "ModuleA"
    assert mod["final_review"] == "reviewed"
    assert len(mod["children"]) == 1
    cls = mod["children"][0]
    assert cls["name"] == "ClassA"
    assert cls["final_review"] == "reviewed"
    assert len(cls["children"]) == 1
    fn = cls["children"][0]
    assert fn["name"] == "funcA"
    assert fn["final_output"] == fn_code
    assert fn["final_review"] == "reviewed"
    assert result["error"] is False


def test_orchestrate_full_workflow_empty_requirements(monkeypatch, tmp_path):
    class DummyTask:
        agent = object()
        output = type("Out", (), {"pydantic": None, "raw": "[]"})()

    class DummyReview:
        agent = object()
        output = type("Out", (), {"raw_output": "reviewed"})()

    monkeypatch.setattr(
        "convoke.workflow.create_architect_task", lambda req: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_architect_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_module_manager_task", lambda n, d: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_module_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_class_manager_task", lambda n, d: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_class_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_function_manager_task", lambda n, d: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_function_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.write_project_scaffolding", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "convoke.workflow.write_tree_visualization", lambda *a, **k: None
    )
    logger = logging.getLogger("test")
    result = orchestrate_full_workflow(
        requirements="",
        max_depth=1,
        max_items=1,
        max_tasks=10,
        verbose=0,
        logger=logger,
        review_cycles=1,
        parse_retries=1,
        output_dir=str(tmp_path),
        artifact_store=DummyArtifactStore(),
        get_tool=None,
        save_tool=None,
    )
    assert result["modules"] == []
    assert result["error"] is False


def test_orchestrate_level_max_depth():
    logger = logging.getLogger("test")

    def dummy_task_fn(name, desc):
        class DummyTask:
            agent = object()
            output = type("Out", (), {"pydantic": None, "raw": "[]"})()

        return DummyTask()

    def dummy_review_fn(task):
        class DummyReview:
            agent = object()
            output = type("Out", (), {"raw_output": "reviewed"})()

        return DummyReview()

    def dummy_refine_fn(*a, **k):
        return dummy_task_fn(a[1], a[2])

    def dummy_parse_fn(text, max_items):
        return []

    results = orchestrate_level(
        [("Parent", "desc")],
        dummy_task_fn,
        dummy_review_fn,
        dummy_refine_fn,
        dummy_parse_fn,
        10,  # recursion_depth
        2,  # max_depth
        2,  # max_items
        logger,
        10,  # max_tasks
        0,  # verbose
        1,  # review_cycles
        1,  # parse_retries
    )
    assert results == []


def test_orchestrate_level_max_tasks():
    logger = logging.getLogger("test")

    def dummy_task_fn(name, desc):
        class DummyTask:
            agent = object()
            output = type("Out", (), {"pydantic": None, "raw": "[]"})()

        return DummyTask()

    def dummy_review_fn(task):
        class DummyReview:
            agent = object()
            output = type("Out", (), {"raw_output": "reviewed"})()

        return DummyReview()

    def dummy_refine_fn(*a, **k):
        return dummy_task_fn(a[1], a[2])

    def dummy_parse_fn(text, max_items):
        return []

    # Set task_counter high to trigger cap
    import convoke.workflow

    convoke.workflow.task_counter["count"] = 100
    results = orchestrate_level(
        [("Parent", "desc")],
        dummy_task_fn,
        dummy_review_fn,
        dummy_refine_fn,
        dummy_parse_fn,
        1,  # recursion_depth
        2,  # max_depth
        2,  # max_items
        logger,
        100,  # max_tasks
        0,  # verbose
        1,  # review_cycles
        1,  # parse_retries
    )
    assert results == []
    convoke.workflow.task_counter["count"] = 0  # reset for other tests


def test_orchestrate_full_workflow_error_propagation(monkeypatch, tmp_path):
    class DummyTask:
        agent = object()
        output = type("Out", (), {"pydantic": None, "raw": "[]"})()

    class DummyReview:
        agent = object()
        output = type("Out", (), {"raw_output": "reviewed"})()

    def error_task(*a, **k):
        return DummyTask()

    def error_review(*a, **k):
        return DummyReview()

    monkeypatch.setattr(
        "convoke.workflow.create_architect_task", lambda req: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_architect_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_module_manager_task", lambda n, d: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_module_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_class_manager_task", lambda n, d: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_class_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_function_manager_task", lambda n, d: DummyTask()
    )
    monkeypatch.setattr(
        "convoke.workflow.create_function_review_task", lambda t: DummyReview()
    )
    monkeypatch.setattr(
        "convoke.workflow.write_project_scaffolding", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "convoke.workflow.write_tree_visualization", lambda *a, **k: None
    )
    # Patch run_task_with_review_and_refine to always return error
    import convoke.workflow

    monkeypatch.setattr(
        convoke.workflow,
        "run_task_with_review_and_refine",
        lambda *a, **k: {"error": True, "error_msg": "fail"},
    )
    logger = logging.getLogger("test")
    result = orchestrate_full_workflow(
        requirements="reqs",
        max_depth=1,
        max_items=1,
        max_tasks=10,
        verbose=0,
        logger=logger,
        review_cycles=1,
        parse_retries=1,
        output_dir=str(tmp_path),
        artifact_store=DummyArtifactStore(),
        get_tool=None,
        save_tool=None,
    )
    assert result["error"] is True
    assert "fail" in result["error_msg"]

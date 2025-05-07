# Refactoring Plan for `convoke/convoke.py`

This refactoring breaks apart the monolithic `convoke/convoke.py` into cohesive modules, improving readability, testability, and maintainability.

## 1. CLI Layer (`convoke/cli.py`)

- Move `if __name__ == '__main__': main()` and all argument parsing into `cli.py`.
- Responsibilities:
  - Parse command-line args (`--project-path`, `--max-depth`, etc).
  - Load project config (`config.yaml`).
  - Instantiate `FileSystemArtifactStore` and scoped tools.
  - Call `orchestrate_full_workflow` from orchestrator.
  - Set up logging and signal handlers.
  - Print summary and exit codes.

## 2. Agents & Tasks Factory (`convoke/agents.py`)

- Extract all `create_*_agent` and `create_*_task` functions.
- Keep definitions of `create_architect_agent`, `create_module_manager_agent`, etc.
- Export public API for orchestrator to import.

## 3. Parsing Utilities (`convoke/parsing.py`)

- Move `parse_numbered_list` and `parse_json_list` here.
- Include any regex or helper logic.
- Write small unit tests for edge cases.

## 4. Core Orchestration (`convoke/orchestrator.py`)

- Consolidate:
  - `run_task_with_review` and `run_task_with_review_and_refine`.
  - `orchestrate_level` and its `make_next_level_handler`.
  - `orchestrate_full_workflow` (architecture through functions).
- Dependencies:
  - Import agent/task factories from `agents.py`.
  - Import parsing utils from `parsing.py`.
  - Accept `artifact_store`, `get_tool`, `save_tool` from CLI.
- Return a result dict; no direct I/O aside from tool calls.

## 5. Artifact Store & Tools (`convoke/store.py`, `convoke/tools.py`)

- Already implemented; just update imports and ensure types.
- Ensure `BaseTool` stub remains in `crewai_tools.py` or move to a new `tools/base.py`.

## 6. I/O Helpers (`convoke/io.py`)

- Extract:
  - `lint_python_code` and any subprocess calls.
  - `write_project_scaffolding` and `write_tree_visualization`.
- Keep `ensure_dir_exists`, `write_json_file`, `write_text_file` (deprecated in favor of store?) for backwards compat.

## 7. Utilities (`convoke/utils.py`)

- Extract:
  - `setup_logging` and `ensure_api_keys`.
  - Common constants or helper functions.

## 8. Test Updates

- Update `tests/` to reflect new module paths:
  - Tests for store and tools remain.
  - Add tests for parsing utils (`parsing.py`).
  - Add tests for CLI (`cli.py`) using `subprocess` or `pytest` fixtures.
  - Write integration test for `orchestrate_full_workflow` with a stubbed store.

## 9. Deprecation & Entry Points

- Leave `convoke/convoke.py` as a thin shim:

  ```python
  from convoke.cli import main
  if __name__ == '__main__': main()
  ```

- Update `pyproject.toml` console script to point to `convoke.cli:main`.
- Remove direct imports of `convoke/convoke.py` in docs or scripts.

## 10. Lint, Format & Type Check

- Run `mypy` on the new modules; fix type hints and imports.
- Enforce `flake8` and `black` formatting across new files.
- Ensure no cyclic imports.

---

Following this plan will transform the codebase into clear layers:

```
convoke/
├── agents.py
├── cli.py
├── io.py
├── orchestrator.py
├── parsing.py
├── store.py
├── tools.py
├── utils.py
└── __init__.py
```

Each module has a focused responsibility, making future enhancements and debugging easier.  

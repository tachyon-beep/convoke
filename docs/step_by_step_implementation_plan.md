# Step-by-Step Implementation Plan for Modular Convoke System

This plan details the concrete steps to implement and maintain a clean, modular, and testable Convoke codebase, following the refactoring and design principles in previous documentation.

---

## 1. Directory & Module Structure

- Ensure all core logic is inside the `convoke/` package.
- Remove any root-level modules (e.g., `utils.py`, `workflow.py`) that are not part of the package.
- The following modules should exist in `convoke/`:
  - `agents.py` — Agent and task factories
  - `parsers.py` — Output parsing utilities
  - `project.py` — Project output helpers (README, tree, etc.)
  - `store.py` — Artifact store abstraction
  - `tools.py` — Scoped artifact tools
  - `crewai_tools.py` — BaseTool stub
  - `utils.py` — Logging, API key checks, and other cross-cutting helpers
  - `workflow.py` — Orchestration logic (run_task_with_review, orchestrate_full_workflow, etc.)
  - `cli.py` — CLI entry point

## 2. Implementation Steps

### 2.1. Utilities

- Place `setup_logging` and `ensure_api_keys` in `convoke/utils.py`.
- Update all imports to use `from convoke.utils import ...`.

### 2.2. Workflow Logic

- Place all orchestration logic (task/review/refine, recursive orchestration, etc.) in `convoke/workflow.py`.
- Ensure all workflow functions import agents, parsers, and project helpers from the correct convoke modules.
- Expose a single entry point: `orchestrate_full_workflow`.

### 2.3. CLI

- The CLI should live in `convoke/cli.py`.
- It should:
  - Parse arguments
  - Load config
  - Set up logging and API keys
  - Instantiate the artifact store and tools
  - Call `orchestrate_full_workflow` from `convoke.workflow`
  - Print or log results

### 2.4. Agents, Parsers, Project Output

- `agents.py`: All agent and task creation logic
- `parsers.py`: All output parsing and validation helpers
- `project.py`: All project output helpers (README, tree, lint, etc.)

### 2.5. Artifact Store & Tools

- `store.py`: FileSystemArtifactStore
- `tools.py`: ScopedGetArtifactTool, ScopedSaveArtifactTool
- `crewai_tools.py`: BaseTool stub

### 2.6. Tests

- Place all tests in `tests/`.
- Add/maintain unit tests for store, tools, and (if possible) workflow logic.
- Use `tests/conftest.py` to ensure the project root is on `sys.path` for imports.

### 2.7. Documentation

- Update `docs/implementation_plan.md` and `docs/refactoring_plan.md` as needed.
- Add this step-by-step plan as `docs/step_by_step_implementation_plan.md`.

---

## 3. Verification & Maintenance

- After each refactor, run `pytest` to ensure all tests pass.
- Run `mypy` and `ruff` to check types and style.
- Ensure all CLI and module imports work as intended (no ImportError).
- Keep all new logic inside the `convoke/` package for maintainability.

---

## 4. Example Workflow

1. User runs `python -m convoke.cli --project-path projects/SomeProject_YYYYMMDD_HHMMSS`
2. CLI loads config, sets up logging, and calls `orchestrate_full_workflow`.
3. Workflow logic coordinates agents, tasks, and artifact storage.
4. All outputs are written to the correct project directory via the artifact store and project helpers.
5. Results and logs are printed or saved as appropriate.

---

**This plan ensures a robust, modular, and maintainable codebase for the Convoke system.**

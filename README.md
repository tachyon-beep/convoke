# Convoke

Convoke is a modular, testable, and extensible workflow orchestration system designed for AI agent collaboration, artifact management, and project automation.

## Features

- Modular architecture with clear separation of concerns
- Agent/task orchestration and review workflows
- Artifact storage and retrieval
- CLI for easy project execution
- Comprehensive test suite

## Directory Structure

```
convoke/           # Core package
  agents.py        # Agent and task factories
  cli.py           # CLI entry point
  crewai_tools.py  # BaseTool stub
  project.py       # Project output helpers (README, tree, etc.)
  store.py         # Artifact store abstraction
  tools.py         # Scoped artifact tools
  utils.py         # Logging, API key checks, etc.
  workflow.py      # Orchestration logic
  config.yaml      # Example config

projects/          # Project outputs and artifacts
scripts/           # Utility scripts
  setup_new_project.py

docs/              # Documentation
  step_by_step_implementation_plan.md
  production_web_interface_plan.md

tests/             # Unit tests
  test_project.py
  test_store.py
  test_tools.py
  test_workflow.py
  conftest.py

pyproject.toml      # Poetry project config
poetry.lock         # Poetry lock file
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repo-url>
   cd convoke
   ```

2. Install dependencies with Poetry:

   ```bash
   poetry install
   ```

## Usage

To run a workflow for a project:

```bash
python -m convoke.cli --project-path projects/SomeProject_YYYYMMDD_HHMMSS
```

- The CLI loads config, sets up logging, and calls the workflow orchestrator.
- Results and logs are saved in the specified project directory.

## Development

- All core logic is inside the `convoke/` package.
- Add new features as modules within `convoke/`.
- Run tests with:

  ```bash
  poetry run pytest
  ```

- Check types and style with:

  ```bash
  poetry run mypy convoke/
  poetry run ruff check convoke/
  ```

## Testing

- All tests are in the `tests/` directory.
- Use `conftest.py` to ensure correct import paths.

## Documentation

- See `docs/step_by_step_implementation_plan.md` for the implementation plan.
- See `docs/production_web_interface_plan.md` for future web interface plans.

## License

Specify your license here.

---

**Convoke** — Modular AI workflow orchestration made easy.

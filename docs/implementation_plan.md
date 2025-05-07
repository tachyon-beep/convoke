# Implementation Plan: Refined Project Structure & Workflow for Convoke

This document outlines a step-by-step plan to refactor and extend the Convoke system, enabling isolated project runs, per-project configuration management, and scoped artifact access for agents.

## 1. Directory Structure & Project Initialization

1. Move orchestration logic into `convoke/basic.py` and remove direct file writes from `convoke/convoke.py`.
2. Add a `setup_new_project.py` script at the repo root to:
   - Create a new project folder under `projects/{project_id}/`.
   - Copy `convoke/config_template.yaml` into the new folder as `config.yaml`.
   - Save initial requirements to `initial_requirements.md` (optional).
3. Ensure each project folder contains:
   - `config.yaml`
   - `initial_requirements.md` (optional)
   - `output/` (artifact store root)
   - `logs/` (optional)

## 2. Configuration Management

- Create `convoke/config_template.yaml` as the base configuration.
- On project setup, merge user-specified requirements into the copied `config.yaml`.
- In `basic.py`, load the project-specific `config.yaml` to configure max depths, agent models, etc.

## 3. Implement FileSystemArtifactStore

- New class `FileSystemArtifactStore(base_path, logger)` with methods:
  - `_resolve_path(relative_path)`: normalize and prevent path traversal.
  - `save_artifact(relative_path, content, is_json=False)`.
  - `get_artifact(relative_path)`.
  - `list_artifacts(dir_relative_path) -> List[str]`.
- Replace calls to `write_text_file` and `write_json_file` in orchestration with store methods.

## 4. Scoped Artifact Tools

- Implement two new tools inheriting from `BaseTool`:
  1. `ScopedGetArtifactTool`: checks `allowed_read_prefixes` before calling `store.get_artifact()`.
  2. `ScopedSaveArtifactTool`: checks `allowed_write_prefixes` before calling `store.save_artifact()`.
- When instantiating agents for modules, classes, or functions, calculate their scope prefixes and pass them to the tools.

## 5. Refactor Orchestration (`basic.py`)

1. Add a `--project_path` CLI argument. Deprecate `--requirements` flag.
2. In `main()`, derive `output_dir = os.path.join(project_path, 'output')`.
3. Instantiate `artifact_store = FileSystemArtifactStore(output_dir, logger)`.
4. Replace all direct file operations in `orchestrate_full_workflow`:
   - Save architecture JSON and review via the store.
   - In `save_module_tree()`, use scoped save tool or store to write module/class/function artifacts.
   - In `write_project_scaffolding()` and `write_tree_visualization()`, direct writes remain, but target `output_dir`.

## 6. Agent Initialization & Tool Injection

- In `basic.py`, when building the Architect agent, pass it a `ScopedSaveArtifactTool` with write prefix `""` or `["system_architecture.json"]`.
- For Module Manager agents, calculate prefix `modules/{module_name}/` and inject get/save tools accordingly.
- Ensure each agent only uses its scoped tools.

## 7. Logging & Isolation

- Modify `setup_logging()` to create per-project log files under `{project_path}/logs/`.
- Ensure all loggers write to both console and project-specific log files.

## 8. Backward Compatibility & Documentation

- Update README with new CLI usage: `python convoke/basic.py --project_path projects/{id}`.
- Provide a migration guide for existing `output/` folders.
- Update docs/production_web_interface_plan.md with pointers to the new structure.

## 9. Testing

- Add unit tests for:
  - `FileSystemArtifactStore` path resolution and safety.
  - `ScopedGetArtifactTool`/`ScopedSaveArtifactTool` permission enforcement.
  - `setup_new_project.py` project directory creation.
- Add integration tests to ensure full run writes artifacts under the correct project folder.

---

**Next Steps:**

1. Implement `setup_new_project.py`.
2. Create `config_template.yaml` under `convoke/`.
3. Develop `FileSystemArtifactStore` in a new module `convoke/store.py`.
4. Build `ScopedGetArtifactTool` and `ScopedSaveArtifactTool` in `convoke/tools.py`.
5. Refactor `convoke/basic.py` to use the new store and tools.
6. Update tests in `tests/` to cover new components.

# Improved Agent Output Handling Plan

This document outlines a comprehensive plan to improve how different agent levels (architecture, module, class, function) handle and exchange written products in the Convoke system.

## Problem Statement

The current workflow system has the following issues:

1. **Inconsistent Output Handling**: Different types of agents produce different types of outputs (JSON lists vs Python code), but they're all processed through the same workflow logic with ad-hoc handling.

2. **Limited Metadata**: Artifacts don't maintain consistent metadata or parent-child relationships.

3. **Disjointed Storage**: Code and metadata are saved inconsistently, with some duplicated in both artifact store and output directory.

4. **Lack of Validation**: Function code doesn't undergo consistent validation or standardization.

5. **Ineffective Review Integration**: Reviews aren't properly integrated with the artifacts they correspond to.

## Architecture Improvements

### 1. Unified Artifact Model

Create a standardized metadata model for all artifacts:

```python
class ArtifactMetadata:
    id: str                      # Unique ID for the artifact
    name: str                    # Name of the component (module, class, function)
    description: str             # Description of the component
    type: str                    # "architecture", "module", "class", "function", "test"
    content_type: str            # "json", "python", "markdown", etc.
    created_at: str              # ISO timestamp
    path: str                    # File path fragment (name_cleaned)
    parent_id: Optional[str]     # ID of parent artifact
    parent_type: Optional[str]   # Type of parent artifact
    full_path: str               # Full path including parents
```

### 2. Type-Specific Output Processors

Implement a processor hierarchy to handle different types of outputs consistently:

```
OutputProcessor (base class)
├── JsonListOutputProcessor (for architecture, module, class)
├── PythonCodeOutputProcessor (for functions)
└── TestCodeOutputProcessor (for tests)
```

Each processor will:

- Validate the output format
- Apply standardization
- Save to artifact store (and optionally output directory)
- Return success/error status

### 3. Artifact Registry

Add a central registry to track all artifacts and their relationships:

- Track every artifact by ID
- Maintain parent-child relationships
- Support querying by type, parent, etc.
- Generate full artifact tree

### 4. Improved Integration Points

#### Workflow Integration

Update the workflow orchestration to:

1. Create appropriate metadata for each artifact
2. Use the right processor for each artifact type
3. Handle and propagate errors consistently

#### Tool Integration

Enhance the artifact tools:

1. Support retrieving artifacts by ID or path
2. Allow getting related artifacts (parent, children)
3. Support saving with proper metadata

## Implementation Steps

### Phase 1: Core Components

1. **Create the `artifact_manager.py` module with:**
   - `ArtifactMetadata` Pydantic model
   - `OutputProcessor` and implementations
   - `ArtifactRegistry` class
   - `OutputManager` to coordinate processors and registry

2. **Add utility functions for:**
   - Code validation and standardization
   - Docstring enforcement
   - Integration with linting

### Phase 2: Workflow Integration

1. **Update `workflow.py` to:**
   - Create and use `OutputManager` instances
   - Generate proper metadata for each level
   - Process outputs using appropriate processors
   - Track artifact relationships

2. **Modify handler generation to:**
   - Pass artifact metadata down the handler chain
   - Use type-specific processing logic
   - Return consistent result structures

### Phase 3: Output and Storage Improvements

1. **Enhance artifact storage:**
   - Structured directory layout
   - Consistent file naming
   - Proper separation of code and metadata

2. **Improve output directory organization:**
   - Mirror logical hierarchy
   - Generate proper import paths
   - Create package structure

## Detailed Implementation

### 1. New `artifact_manager.py` File

This file will contain:

- `ArtifactMetadata` model for standardized metadata
- `OutputProcessor` base class and implementations:
  - `JsonListOutputProcessor` for JSON list outputs
  - `PythonCodeOutputProcessor` for function code
  - `TestCodeOutputProcessor` for test code
- `ArtifactRegistry` to track all artifacts
- `OutputManager` to coordinate processing

### 2. Updates to `workflow.py`

Modify:

- `make_next_level_handler` to use artifact metadata and output processors
- `run_task_with_review_and_refine` to create and pass metadata
- `orchestrate_level` to track parent-child relationships
- `orchestrate_full_workflow` to initialize the artifact registry

### 3. Updates to `store.py`

Enhance:

- `FileSystemArtifactStore` to work with the new metadata structure
- Add methods to query artifacts by relationships
- Support versioning and lifecycle changes

### 4. Updates to Tools

Modify:

- `tools.py` to integrate with the artifact registry
- Add new tool functions for relationship queries
- Support metadata-driven artifact retrieval

## Benefits

This refactoring will:

1. **Improve Consistency**: All outputs are processed using a consistent pattern
2. **Enhance Traceability**: Maintain clear parent-child relationships
3. **Better Validation**: Validate and standardize all outputs
4. **Simplified Integration**: Cleaner integration with storage and output
5. **Improved Review Flow**: Reviews are directly tied to artifacts

## Potential Extensions

Future improvements could include:

1. **Versioning**: Track changes to artifacts over multiple runs
2. **Diff Support**: Show differences between versions
3. **Review Integration**: Apply review comments automatically
4. **Import Inference**: Automatically generate import statements
5. **Interactive Editor**: Web-based editing of artifacts
6. **Dependency Tracking**: Track dependencies between components

## Example Usage

```python
# Initialize components
registry = ArtifactRegistry()
output_manager = OutputManager(artifact_store, output_dir, registry)

# During workflow execution:
# 1. Create metadata for a module
module_metadata = output_manager.create_metadata(
    name="User Manager",
    description="Handles user authentication and profiles",
    artifact_type="module",
    content_type="json"
)

# 2. Process module output
success, error, updated_metadata = output_manager.process_output(
    content='[{"name": "User", "description": "User model class"}]',
    review="Looks good, needs error handling",
    metadata=module_metadata
)

# 3. Create child artifact (class)
class_metadata = output_manager.create_metadata(
    name="User",
    description="User model class",
    artifact_type="class",
    content_type="json",
    parent=module_metadata
)

# Process continues for functions...
```

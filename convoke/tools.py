import logging
from typing import List, Optional, Any
from convoke.store import FileSystemArtifactStore
from crewai.tools import tool
import os


# Initialize the artifact store with proper configuration
# Use a project-specific path if available, otherwise default to ./artifacts
default_artifacts_path = os.environ.get("CONVOKE_ARTIFACTS_DIR", "./artifacts")
artifact_store = FileSystemArtifactStore(
    base_path=default_artifacts_path, logger=logging.getLogger(__name__)
)


@tool("GetProjectArtifact")
def get_project_artifact(
    artifact_path: str,
    agent_role: str = None,
    allowed_read_prefixes: List[str] = None,
    store: Any = None,
) -> str:
    """Retrieves a project artifact within allowed read prefixes."""
    # Use the provided store or fall back to the global artifact_store
    actual_store = None

    # If store is provided and looks valid, use it
    if store is not None and hasattr(store, "get_artifact"):
        actual_store = store
    else:
        # Fall back to the global artifact_store
        actual_store = artifact_store

    if actual_store is None:
        return f"Error: No artifact store available"

    try:
        content = actual_store.get_artifact(artifact_path)
        if content is None:
            return f"Error: Artifact '{artifact_path}' not found"
        return content
    except PermissionError:
        return f"Error: Access denied to artifact '{artifact_path}'"
    except Exception as e:
        return f"Error retrieving artifact: {str(e)}"


@tool("SaveProjectArtifact")
def save_project_artifact(
    artifact_path: str,
    content: str,
    is_json_content: bool,
    agent_role: str = None,
    module_name: str = None,
    class_name: str = None,
    allowed_write_prefixes: List[str] = None,
    store: Any = None,
) -> str:
    """Saves a project artifact within allowed write prefixes.

    Args:
        artifact_path: The path to save the artifact to
        content: The content to save
        is_json_content: Whether the content is JSON
        agent_role: The role of the agent saving the artifact
        module_name: Optional module name for prefixing filenames
        class_name: Optional class name for prefixing filenames
        allowed_write_prefixes: List of allowed write prefixes
        store: Optional artifact store to use

    Returns:
        Success message or error message
    """
    # Use the provided store or fall back to the global artifact_store
    actual_store = None

    # If store is provided and looks valid, use it
    if store is not None and hasattr(store, "save_artifact"):
        actual_store = store
    else:
        # Fall back to the global artifact_store
        actual_store = artifact_store

    if actual_store is None:
        return f"Error: No artifact store available"

    # Create a more descriptive path if module or class context is provided
    # This prevents different modules from overwriting each other's artifacts
    if module_name or class_name:
        path_parts = os.path.split(artifact_path)
        filename = path_parts[1]
        directory = path_parts[0] if len(path_parts) > 1 else ""

        # If we're saving a module design and module_name is provided
        if "module_design" in filename and module_name:
            safe_module_name = module_name.replace(" ", "_")
            if not filename.startswith(safe_module_name):
                filename = f"{safe_module_name}_{filename}"

        # If we're saving a class design and class_name is provided
        elif "class_design" in filename and class_name:
            safe_class_name = class_name.replace(" ", "_")
            if not filename.startswith(safe_class_name):
                filename = f"{safe_class_name}_{filename}"

        # If module_name is provided but not in filename, add it as a directory
        elif module_name and "module" not in filename and "system" not in filename:
            safe_module_name = module_name.replace(" ", "_")
            directory = (
                os.path.join(directory, safe_module_name)
                if directory
                else safe_module_name
            )

        # Reconstruct the path
        artifact_path = os.path.join(directory, filename) if directory else filename

    try:
        result = actual_store.save_artifact(
            artifact_path, content, is_json=is_json_content
        )
        return result
    except PermissionError:
        return f"Error: Write access denied to path '{artifact_path}'"
    except Exception as e:
        return f"Error saving artifact: {str(e)}"


# Export the tool objects directly
scoped_get_artifact = get_project_artifact
scoped_save_artifact = save_project_artifact

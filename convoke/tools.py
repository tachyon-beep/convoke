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
    allowed_write_prefixes: List[str] = None,
    store: Any = None,
) -> str:
    """Saves a project artifact within allowed write prefixes."""
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

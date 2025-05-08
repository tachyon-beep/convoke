import logging
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from convoke.store import FileSystemArtifactStore
from crewai.tools import tool


# Add a global configuration for Pydantic to allow arbitrary types
class ToolConfig:
    model_config = ConfigDict(arbitrary_types_allowed=True)


@tool("GetProjectArtifact")
def scoped_get_artifact(
    artifact_path: str,
    agent_role: str,
    allowed_read_prefixes: List[str],
    store: FileSystemArtifactStore,
) -> str:
    """Retrieves a project artifact within allowed read prefixes."""
    normalized = artifact_path.lstrip("/")
    for prefix in allowed_read_prefixes:
        if normalized.startswith(prefix.lstrip("/")):
            return store.get_artifact(normalized)
    store.logger.warning(
        f"Access DENIED for role {agent_role} to read '{artifact_path}'"
    )
    return f"Error: Access denied to artifact '{artifact_path}'"


@tool("SaveProjectArtifact")
def scoped_save_artifact(
    artifact_path: str,
    content: str,
    is_json_content: bool,
    agent_role: str,
    allowed_write_prefixes: List[str],
    store: FileSystemArtifactStore,
) -> str:
    """Saves a project artifact within allowed write prefixes."""
    normalized = artifact_path.lstrip("/")
    for prefix in allowed_write_prefixes:
        if normalized.startswith(prefix.lstrip("/")):
            return store.save_artifact(normalized, content, is_json=is_json_content)
    store.logger.warning(
        f"Access DENIED for role {agent_role} to write '{artifact_path}'"
    )
    return f"Error: Write access denied to path '{artifact_path}'"


# Updated `scoped_get_artifact` and `scoped_save_artifact` to be callable by wrapping them in a class.


class ScopedGetArtifact:
    def __init__(self, store, agent_role, allowed_read_prefixes):
        self.store = store
        self.agent_role = agent_role
        self.allowed_read_prefixes = allowed_read_prefixes

    def __call__(self, artifact_path):
        normalized = artifact_path.lstrip("/")
        for prefix in self.allowed_read_prefixes:
            if normalized.startswith(prefix.lstrip("/")):
                return self.store.get_artifact(normalized)
        self.store.logger.warning(
            f"Access DENIED for role {self.agent_role} to read '{artifact_path}'"
        )
        return f"Error: Access denied to artifact '{artifact_path}'"


class ScopedSaveArtifact:
    def __init__(self, store, agent_role, allowed_write_prefixes):
        self.store = store
        self.agent_role = agent_role
        self.allowed_write_prefixes = allowed_write_prefixes

    def __call__(self, artifact_path, content, is_json_content):
        normalized = artifact_path.lstrip("/")
        for prefix in self.allowed_write_prefixes:
            if normalized.startswith(prefix.lstrip("/")):
                return self.store.save_artifact(
                    normalized, content, is_json=is_json_content
                )
        self.store.logger.warning(
            f"Access DENIED for role {self.agent_role} to write '{artifact_path}'"
        )
        return f"Error: Write access denied to path '{artifact_path}'"

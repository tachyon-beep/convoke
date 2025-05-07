import logging
from pydantic import BaseModel, Field
from typing import List, Optional
from convoke.crewai_tools import BaseTool
from convoke.store import FileSystemArtifactStore


class ScopedGetArtifactTool(BaseTool):
    """Retrieves a project artifact within allowed read prefixes."""

    name = "GetProjectArtifact"
    description = (
        "Retrieves a project artifact. Provide a relative path from output root."
    )
    # these must be set when instantiating the tool
    store: FileSystemArtifactStore
    agent_role: str
    allowed_read_prefixes: List[str]

    class ArgsSchema(BaseModel):
        artifact_path: str = Field(
            description="Relative artifact path from project output root"
        )

    def _run(self, artifact_path: str) -> Optional[str]:
        normalized = artifact_path.lstrip("/")
        for prefix in self.allowed_read_prefixes:
            if normalized.startswith(prefix.lstrip("/")):
                return self.store.get_artifact(normalized)
        self.logger = self.store.logger
        self.logger.warning(
            f"Access DENIED for role {self.agent_role} to read '{artifact_path}'"
        )
        return f"Error: Access denied to artifact '{artifact_path}'"


class ScopedSaveArtifactTool(BaseTool):
    """Saves a project artifact within allowed write prefixes."""

    name = "SaveProjectArtifact"
    description = (
        "Saves a project artifact. Provide relative path, content, and is_json flag."
    )
    store: FileSystemArtifactStore
    agent_role: str
    allowed_write_prefixes: List[str]

    class ArgsSchema(BaseModel):
        artifact_path: str = Field(
            description="Relative artifact path from project output root"
        )
        content: str = Field(description="Content to save as artifact")
        is_json_content: bool = Field(
            False,
            description="True if content is JSON-serializable dict/list or JSON string",
        )

    def _run(
        self, artifact_path: str, content: str, is_json_content: bool = False
    ) -> str:
        normalized = artifact_path.lstrip("/")
        for prefix in self.allowed_write_prefixes:
            if normalized.startswith(prefix.lstrip("/")):
                return self.store.save_artifact(
                    normalized, content, is_json=is_json_content
                )
        self.logger = self.store.logger
        self.logger.warning(
            f"Access DENIED for role {self.agent_role} to write '{artifact_path}'"
        )
        return f"Error: Write access denied to path '{artifact_path}'"

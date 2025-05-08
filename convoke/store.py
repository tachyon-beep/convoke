import os
import json
import logging
from typing import Optional, List, Union


class FileSystemArtifactStore:
    """Filesystem-backed artifact store rooted at a base directory."""

    def __init__(
        self,
        base_path: str,
        logger: Optional[logging.Logger] = None,
        allowed_read_prefixes: Optional[List[str]] = None,
        allowed_write_prefixes: Optional[List[str]] = None,
        agent_role: str = "default",
    ) -> None:
        self.base_path = os.path.abspath(base_path)
        self.logger = logger or logging.getLogger(__name__)
        self.allowed_read_prefixes = allowed_read_prefixes or [
            ""
        ]  # Empty string means "all"
        self.allowed_write_prefixes = allowed_write_prefixes or [
            ""
        ]  # Empty string means "all"
        self.agent_role = agent_role
        os.makedirs(self.base_path, exist_ok=True)

    def create_child_store(
        self, scope_path: str, agent_role: str, inherit_read: bool = True
    ) -> "FileSystemArtifactStore":
        """
        Create a child store with the same base_path but restricted access permissions.

        Args:
            scope_path: The path prefix for this agent's primary scope
            agent_role: The role of the agent using this store
            inherit_read: Whether to inherit parent's read permissions

        Returns:
            A new FileSystemArtifactStore with appropriate access restrictions
        """
        # Normalize scope path
        scope_path = scope_path.lstrip("/")

        # Set up allowed write prefixes - only the agent's scope
        write_prefixes = [scope_path] if scope_path else [""]

        # Set up allowed read prefixes - ALWAYS inherit ALL parent read permissions
        # This ensures that files from ancestors are always visible to descendants
        read_prefixes = (
            list(self.allowed_read_prefixes) if self.allowed_read_prefixes else []
        )

        # Always include agent's own scope if not already included
        if scope_path and scope_path not in read_prefixes:
            read_prefixes.append(scope_path)

        # Remove any duplicates while preserving order
        unique_read_prefixes = []
        for prefix in read_prefixes:
            if prefix not in unique_read_prefixes:
                unique_read_prefixes.append(prefix)

        # Create child store with the same base_path but updated access controls
        return FileSystemArtifactStore(
            base_path=self.base_path,
            logger=self.logger,
            allowed_read_prefixes=unique_read_prefixes,
            allowed_write_prefixes=write_prefixes,
            agent_role=agent_role,
        )

    def _check_read_permission(self, relative_path: str) -> bool:
        """Check if the current store has permission to read a path"""
        normalized = relative_path.lstrip("/")

        # Check against allowed read prefixes
        for prefix in self.allowed_read_prefixes:
            prefix = prefix.lstrip("/")
            # Empty prefix means "all access"
            if prefix == "" or normalized.startswith(prefix):
                return True

        self.logger.warning(
            f"Access DENIED for role '{self.agent_role}' to read '{relative_path}'"
        )
        return False

    def _check_write_permission(self, relative_path: str) -> bool:
        """Check if the current store has permission to write to a path"""
        normalized = relative_path.lstrip("/")

        # Check against allowed write prefixes
        for prefix in self.allowed_write_prefixes:
            prefix = prefix.lstrip("/")
            # Empty prefix means "all access"
            if prefix == "" or normalized.startswith(prefix):
                return True

        self.logger.warning(
            f"Access DENIED for role '{self.agent_role}' to write '{relative_path}'"
        )
        return False

    def _resolve_path(self, relative_artifact_path: str) -> str:
        # Normalize and prevent path traversal
        parts = [p for p in relative_artifact_path.split("/") if p and p != "."]
        safe_rel = os.path.join(*parts) if parts else ""
        full_path = os.path.abspath(os.path.join(self.base_path, safe_rel))
        if not full_path.startswith(self.base_path):
            self.logger.error(
                f"Path traversal attempt: '{relative_artifact_path}' -> '{full_path}' outside of base '{self.base_path}'"
            )
            raise ValueError("Artifact path is outside the allowed base directory.")
        return full_path

    def save_artifact(
        self,
        relative_artifact_path: str,
        content: Union[str, dict, list],
        is_json: bool = False,
    ) -> str:
        if not self._check_write_permission(relative_artifact_path):
            raise PermissionError(
                f"Write access denied for path: {relative_artifact_path}"
            )
        full_path = self._resolve_path(relative_artifact_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        if is_json:
            with open(full_path, "w", encoding="utf-8") as f:
                if isinstance(content, (dict, list)):
                    json.dump(content, f, indent=2)
                else:
                    f.write(content)
        else:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
        self.logger.info(f"Artifact saved: {full_path}")
        return f"Artifact saved to {relative_artifact_path}"

    def get_artifact(self, relative_artifact_path: str) -> Optional[str]:
        if not self._check_read_permission(relative_artifact_path):
            raise PermissionError(
                f"Read access denied for path: {relative_artifact_path}"
            )
        full_path = self._resolve_path(relative_artifact_path)
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        self.logger.warning(f"Artifact not found: {full_path}")
        return None

    def list_artifacts(self, dir_relative_path: str = ".") -> List[str]:
        if not self._check_read_permission(dir_relative_path):
            raise PermissionError(f"Read access denied for path: {dir_relative_path}")
        full_dir = self._resolve_path(dir_relative_path)
        if os.path.isdir(full_dir):
            return os.listdir(full_dir)
        return []

    def __get_pydantic_core_schema__(self, handler):
        """Define how Pydantic should handle this type."""
        return handler.generate_schema(str)

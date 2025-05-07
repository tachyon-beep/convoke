import os
import json
import logging
from typing import Optional, List, Union


class FileSystemArtifactStore:
    """Filesystem-backed artifact store rooted at a base directory."""

    def __init__(self, base_path: str, logger: Optional[logging.Logger] = None) -> None:
        self.base_path = os.path.abspath(base_path)
        self.logger = logger or logging.getLogger(__name__)
        os.makedirs(self.base_path, exist_ok=True)

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
        full_path = self._resolve_path(relative_artifact_path)
        if os.path.isfile(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        self.logger.warning(f"Artifact not found: {full_path}")
        return None

    def list_artifacts(self, dir_relative_path: str = ".") -> List[str]:
        full_dir = self._resolve_path(dir_relative_path)
        if os.path.isdir(full_dir):
            return os.listdir(full_dir)
        return []

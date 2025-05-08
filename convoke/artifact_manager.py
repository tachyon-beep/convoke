import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Type
from datetime import datetime
import os
import json
import uuid
import ast
from pydantic import BaseModel


class ArtifactMetadata(BaseModel):
    """Standardized metadata for any artifact in the system."""

    id: str
    name: str
    description: str
    type: str  # "architecture", "module", "class", "function", "test"
    content_type: str  # "json", "python", "markdown", etc.
    created_at: str
    path: str
    parent_id: Optional[str] = None
    parent_type: Optional[str] = None
    full_path: str


# Output processor interface and implementations
class OutputProcessor:
    """Base class for output processors that handle different types of agent outputs."""

    def process_output(
        self,
        content: str,
        review: str,
        metadata: Dict[str, Any],
        artifact_store,
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Process the output from an agent.

        Args:
            content: The content produced by the agent
            review: The review of the content
            metadata: Metadata about the artifact
            artifact_store: The artifact store to save to
            output_dir: Optional output directory for code files

        Returns:
            (success, error_message, processed_metadata)
        """
        raise NotImplementedError("Subclasses must implement process_output")


class JsonListOutputProcessor(OutputProcessor):
    """Processor for JSON list outputs from architecture, module, and class agents."""

    def process_output(
        self,
        content: str,
        review: str,
        metadata: Dict[str, Any],
        artifact_store,
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process and validate JSON list output."""
        try:
            # Validate JSON format
            if not content.strip():
                return False, "Empty content", metadata

            json_data = json.loads(content)

            # Save the JSON content to the artifact store
            artifact_path = f"{metadata['full_path']}.json"
            metadata_path = f"{metadata['full_path']}_metadata.json"
            review_path = f"{metadata['full_path']}_review.txt"

            artifact_store.save_artifact(artifact_path, content)
            artifact_store.save_artifact(metadata_path, json.dumps(metadata, indent=2))

            if review:
                artifact_store.save_artifact(review_path, review)

            return True, "", metadata
        except json.JSONDecodeError as e:
            return False, f"JSON validation error: {str(e)}", metadata
        except Exception as e:
            return False, f"Error processing JSON output: {str(e)}", metadata


class PythonCodeOutputProcessor(OutputProcessor):
    """Processor for Python code outputs from function agents."""

    def process_output(
        self,
        content: str,
        review: str,
        metadata: Dict[str, Any],
        artifact_store,
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process, validate, and format Python code output."""
        try:
            if not content.strip():
                return False, "Empty code content", metadata

            # Validate syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                return False, f"Python syntax error: {str(e)}", metadata

            # Ensure proper docstring if missing
            content = self._ensure_docstring(content, metadata["name"])

            # Run linter if available
            try:
                from convoke.utils import lint_python_code

                lint_result = lint_python_code(content)
                metadata["lint_result"] = lint_result
            except ImportError:
                # Linting not available, that's okay
                pass

            # Save to artifact store
            artifact_path = f"{metadata['full_path']}.py"
            metadata_path = f"{metadata['full_path']}_metadata.json"
            review_path = f"{metadata['full_path']}_review.txt"

            artifact_store.save_artifact(artifact_path, content)
            artifact_store.save_artifact(metadata_path, json.dumps(metadata, indent=2))

            if review:
                artifact_store.save_artifact(review_path, review)

            # Also save to output directory if specified
            if output_dir:
                output_path = os.path.join(output_dir, f"{metadata['path']}.py")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return True, "", metadata
        except Exception as e:
            return False, f"Error processing Python code: {str(e)}", metadata

    def _ensure_docstring(self, code: str, function_name: str) -> str:
        """Ensure the code has a proper docstring."""
        # Simple implementation - could be enhanced
        if '"""' not in code[: code.find("\n\n")]:
            first_line_end = code.find("\n")
            code = (
                code[: first_line_end + 1]
                + f'    """{function_name} function.\n    \n    """\n'
                + code[first_line_end + 1 :]
            )
        return code


class TestCodeOutputProcessor(OutputProcessor):
    """Processor for test code outputs."""

    def process_output(
        self,
        content: str,
        review: str,
        metadata: Dict[str, Any],
        artifact_store,
        output_dir: Optional[str] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process and validate test code output."""
        # Similar to PythonCodeOutputProcessor but with test-specific handling
        try:
            if not content.strip():
                return False, "Empty test content", metadata

            # Validate syntax
            try:
                ast.parse(content)
            except SyntaxError as e:
                return False, f"Test syntax error: {str(e)}", metadata

            # Save to artifact store
            artifact_path = f"{metadata['full_path']}.py"
            metadata_path = f"{metadata['full_path']}_metadata.json"
            review_path = f"{metadata['full_path']}_review.txt"

            artifact_store.save_artifact(artifact_path, content)
            artifact_store.save_artifact(metadata_path, json.dumps(metadata, indent=2))

            if review:
                artifact_store.save_artifact(review_path, review)

            # Also save to output directory if specified
            if output_dir:
                # Tests typically go in a tests/ directory or have test_ prefix
                output_path = os.path.join(output_dir, f"test_{metadata['path']}.py")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)

            return True, "", metadata
        except Exception as e:
            return False, f"Error processing test code: {str(e)}", metadata


# Artifact Registry to track all artifacts
class ArtifactRegistry:
    """Central registry to track all artifacts created during workflow execution."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.artifacts = {}  # id -> metadata
        self.logger = logger or logging.getLogger(__name__)

    def register_artifact(self, metadata: Dict[str, Any]) -> str:
        """Register a new artifact and return its ID."""
        if "id" not in metadata:
            metadata["id"] = str(uuid.uuid4())

        self.artifacts[metadata["id"]] = metadata
        return metadata["id"]

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for an artifact by ID."""
        return self.artifacts.get(artifact_id)

    def get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get all artifacts that have this parent."""
        return [m for m in self.artifacts.values() if m.get("parent_id") == parent_id]

    def get_artifacts_by_type(self, artifact_type: str) -> List[Dict[str, Any]]:
        """Get all artifacts of a specific type."""
        return [m for m in self.artifacts.values() if m.get("type") == artifact_type]

    def get_full_tree(self) -> Dict[str, Any]:
        """Get the full artifact tree organized by hierarchy."""
        result = {}
        # First find all root artifacts (no parent)
        roots = [m for m in self.artifacts.values() if not m.get("parent_id")]

        for root in roots:
            result[root["id"]] = self._build_tree(root["id"])

        return result

    def _build_tree(self, node_id: str) -> Dict[str, Any]:
        """Recursively build a tree of an artifact and all its children."""
        node = self.artifacts.get(node_id, {}).copy()
        children = self.get_children(node_id)

        node["children"] = [self._build_tree(child["id"]) for child in children]
        return node


# Output Manager to handle different types of outputs
class OutputManager:
    """Manager to handle processing different types of agent outputs."""

    def __init__(
        self,
        artifact_store,
        output_dir: Optional[str] = None,
        registry: Optional[ArtifactRegistry] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.artifact_store = artifact_store
        self.output_dir = output_dir
        self.registry = registry or ArtifactRegistry()
        self.logger = logger or logging.getLogger(__name__)

        # Register output processors for different artifact types
        self.processors = {
            "architecture": JsonListOutputProcessor(),
            "module": JsonListOutputProcessor(),
            "class": JsonListOutputProcessor(),
            "function": PythonCodeOutputProcessor(),
            "test": TestCodeOutputProcessor(),
        }

    def create_metadata(
        self,
        name: str,
        description: str,
        artifact_type: str,
        content_type: str,
        parent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized metadata for any artifact."""
        metadata = {
            "id": str(uuid.uuid4()),
            "name": name,
            "description": description,
            "type": artifact_type,  # "architecture", "module", "class", "function", "test"
            "content_type": content_type,  # "json", "python", "markdown", etc.
            "created_at": datetime.now().isoformat(),
            "path": name.replace(" ", "_"),
        }

        if parent:
            metadata["parent_id"] = parent.get("id")
            metadata["parent_type"] = parent.get("type")
            metadata["full_path"] = f"{parent.get('full_path', '')}/{metadata['path']}"
        else:
            metadata["full_path"] = metadata["path"]

        # Register in the artifact registry
        self.registry.register_artifact(metadata)
        return metadata

    def process_output(
        self, content: str, review: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Process an output based on its artifact type."""
        processor = self.processors.get(metadata["type"])
        if not processor:
            error = f"No processor found for artifact type: {metadata['type']}"
            self.logger.error(error)
            return False, error, metadata

        return processor.process_output(
            content, review, metadata, self.artifact_store, self.output_dir
        )

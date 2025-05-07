import os
import tempfile
import pytest
import json
from convoke.store import FileSystemArtifactStore


def test_save_and_get_text_artifact():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileSystemArtifactStore(base_path=tmpdir)
        # Save text artifact
        result = store.save_artifact("folder/file.txt", "hello world", is_json=False)
        assert "Artifact saved to folder/file.txt" in result
        # Get artifact
        content = store.get_artifact("folder/file.txt")
        assert content == "hello world"


def test_save_and_get_json_artifact():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileSystemArtifactStore(base_path=tmpdir)
        data = {"a": 1, "b": [2, 3]}
        result = store.save_artifact("data.json", data, is_json=True)
        assert "Artifact saved to data.json" in result
        content = store.get_artifact("data.json")
        # JSON is returned as formatted string; parse back to dict for comparison
        parsed = json.loads(content) if content is not None else None
        assert parsed == data


def test_list_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileSystemArtifactStore(base_path=tmpdir)
        store.save_artifact("dir1/f1.txt", "x")
        store.save_artifact("dir1/f2.txt", "y")
        files = store.list_artifacts("dir1")
        assert set(files) == {"f1.txt", "f2.txt"}


def test_path_traversal_rejection():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileSystemArtifactStore(base_path=tmpdir)
        # Attempt to save outside base path
        with pytest.raises(ValueError):
            store.save_artifact("../outside.txt", "oops")
        # Attempt to get outside
        with pytest.raises(ValueError):
            store.get_artifact("../../outside.txt")


def test_get_nonexistent_returns_none(caplog):
    with tempfile.TemporaryDirectory() as tmpdir:
        store = FileSystemArtifactStore(base_path=tmpdir)
        content = store.get_artifact("nope.txt")
        assert content is None
        assert "Artifact not found" in caplog.text

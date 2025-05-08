import os
import tempfile
import json
import pytest
from convoke.store import FileSystemArtifactStore
from convoke.tools import scoped_get_artifact, scoped_save_artifact


def setup_store_with_file(tmpdir, rel_path, content):
    store = FileSystemArtifactStore(base_path=tmpdir)
    # save directly via store
    store.save_artifact(rel_path, content, is_json=False)
    return store


def test_scoped_get_allowed(tmp_path):
    tmpdir = str(tmp_path)
    rel = "folder/file.txt"
    content = "test content"
    store = setup_store_with_file(tmpdir, rel, content)
    # Use the _run method for testing, not calling the tool directly
    result = scoped_get_artifact._run(
        artifact_path=rel,
        agent_role="role",
        allowed_read_prefixes=["folder/"],
        store=store,
    )
    assert result == content


def test_scoped_get_denied(tmp_path, caplog):
    tmpdir = str(tmp_path)
    rel = "folder/file.txt"
    content = "test content"
    store = setup_store_with_file(tmpdir, rel, content)
    # Use the _run method for testing, not calling the tool directly
    result = scoped_get_artifact._run(
        artifact_path=rel,
        agent_role="role",
        allowed_read_prefixes=["other/"],
        store=store,
    )
    assert "Error: Access denied" in result
    assert (
        f"Access DENIED for role 'role'" or "Access DENIED for role role" in caplog.text
    )


def test_scoped_save_allowed(tmp_path):
    tmpdir = str(tmp_path)
    store = FileSystemArtifactStore(base_path=tmpdir)
    rel = "save_dir/out.txt"
    content = "hello"
    # Use the _run method for testing, not calling the tool directly
    result = scoped_save_artifact._run(
        artifact_path=rel,
        content=content,
        is_json_content=False,
        agent_role="writer",
        allowed_write_prefixes=["save_dir/"],
        store=store,
    )
    assert "Artifact saved to save_dir/out.txt" in result
    # verify file was created
    path = os.path.join(tmpdir, rel)
    assert os.path.isfile(path)
    with open(path, "r", encoding="utf-8") as f:
        assert f.read() == content


def test_scoped_save_denied(tmp_path, caplog):
    tmpdir = str(tmp_path)
    store = FileSystemArtifactStore(base_path=tmpdir)
    rel = "save_dir/out.txt"
    content = "hello"
    # prefix does not match
    # Use the _run method for testing, not calling the tool directly
    result = scoped_save_artifact._run(
        artifact_path=rel,
        content=content,
        is_json_content=False,
        agent_role="writer",
        allowed_write_prefixes=["other_dir/"],
        store=store,
    )
    assert "Error: Write access denied" in result
    assert (
        f"Access DENIED for role 'writer'"
        or "Access DENIED for role writer" in caplog.text
    )

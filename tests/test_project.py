import os
import tempfile
from convoke.project import write_project_scaffolding, write_tree_visualization
import pytest


def test_write_project_scaffolding_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        modules = [
            {"name": "ModA", "description": "DescA"},
            {"name": "ModB", "description": "DescB"},
        ]
        arch = "System arch text"
        write_project_scaffolding(tmpdir, arch, modules)
        assert os.path.isfile(os.path.join(tmpdir, "README.md"))
        assert os.path.isfile(os.path.join(tmpdir, "requirements.txt"))
        assert os.path.isfile(os.path.join(tmpdir, ".gitignore"))
        with open(os.path.join(tmpdir, "README.md")) as f:
            content = f.read()
            assert "System arch text" in content
            assert "ModA" in content and "ModB" in content


def test_write_tree_visualization_creates_tree_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        modules = [
            {
                "name": "ModA",
                "children": [
                    {
                        "name": "ClassA",
                        "children": [
                            {"name": "funcA"},
                            {"name": "funcB"},
                        ],
                    }
                ],
            }
        ]
        write_tree_visualization(tmpdir, modules)
        tree_path = os.path.join(tmpdir, "PROJECT_TREE.txt")
        assert os.path.isfile(tree_path)
        with open(tree_path) as f:
            content = f.read()
            assert "ModA" in content
            assert "ClassA" in content
            assert "funcA.py" in content and "funcB.py" in content


def test_write_project_scaffolding_empty_modules():
    with tempfile.TemporaryDirectory() as tmpdir:
        modules = []
        arch = "Empty system"
        write_project_scaffolding(tmpdir, arch, modules)
        assert os.path.isfile(os.path.join(tmpdir, "README.md"))
        with open(os.path.join(tmpdir, "README.md")) as f:
            content = f.read()
            assert "Empty system" in content


def test_write_tree_visualization_deeply_nested():
    with tempfile.TemporaryDirectory() as tmpdir:
        modules = [
            {
                "name": "ModA",
                "children": [
                    {
                        "name": "ClassA",
                        "children": [
                            {"name": "funcA"},
                            {"name": "funcB"},
                            {"name": "funcC"},
                        ],
                    },
                    {
                        "name": "ClassB",
                        "children": [
                            {"name": "funcD"},
                        ],
                    },
                ],
            }
        ]
        write_tree_visualization(tmpdir, modules)
        tree_path = os.path.join(tmpdir, "PROJECT_TREE.txt")
        assert os.path.isfile(tree_path)
        with open(tree_path) as f:
            content = f.read()
            assert "ModA" in content
            assert "ClassA" in content and "ClassB" in content
            assert "funcA.py" in content and "funcD.py" in content


def test_write_project_scaffolding_invalid_dir():
    # Should raise an error if directory is invalid
    with pytest.raises(Exception):
        write_project_scaffolding("/invalid/path/should/fail", "arch", [])

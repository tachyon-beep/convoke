import os
from convoke.utils import write_text_file


def write_project_scaffolding(
    output_dir: str, architecture: str, modules: list
) -> None:
    # README.md
    readme_path = os.path.join(output_dir, "README.md")
    module_list = "\n".join(f"- {mod['name']}: {mod['description']}" for mod in modules)
    readme_content = f"""# Generated Project\n\n## System Architecture\n\n{architecture}\n\n## Modules\n\n{module_list}\n\n## How to Run Tests\n\n```bash\npytest\n```\n"""
    write_text_file(readme_path, readme_content)
    # requirements.txt
    requirements_path = os.path.join(output_dir, "requirements.txt")
    requirements_content = "pytest\nflake8\n"
    write_text_file(requirements_path, requirements_content)
    # .gitignore
    gitignore_path = os.path.join(output_dir, ".gitignore")
    gitignore_content = "__pycache__/\n*.pyc\n.env\noutput/\n"
    write_text_file(gitignore_path, gitignore_content)


def write_tree_visualization(output_dir: str, modules: list):
    """Write a text-based tree of the generated project structure."""

    def walk_module_tree(modules, prefix=""):
        lines = []
        for mod in modules:
            mod_name = mod["name"].replace(" ", "_")
            lines.append(f"{prefix}├─ {mod_name}/")
            for cls in mod.get("children", []):
                cls_name = cls["name"].replace(" ", "_")
                lines.append(f"{prefix}│   ├─ {cls_name}/")
                for fn in cls.get("children", []):
                    fn_name = fn["name"].replace(" ", "_")
                    lines.append(f"{prefix}│   │   ├─ {fn_name}.py")
                    lines.append(f"{prefix}│   │   ├─ test_{fn_name}.py")
        return lines

    tree_lines = ["output/"] + walk_module_tree(modules)
    tree_path = os.path.join(output_dir, "PROJECT_TREE.txt")
    write_text_file(tree_path, "\n".join(tree_lines))

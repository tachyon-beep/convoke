#!/usr/bin/env python3
"""
setup_new_project.py

Helper script to create a new Convoke project directory with config and requirements.
"""
import argparse
import os
import shutil
from datetime import datetime


def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Initialize a new Convoke project run")
    parser.add_argument(
        "--name", required=True, help="Project name, e.g. ECommerce_Cart_Feature"
    )
    parser.add_argument(
        "--requirements", required=False, help="High-level requirements text"
    )
    parser.add_argument(
        "--requirements-file",
        required=False,
        help="Path to a file containing requirements",
    )
    args = parser.parse_args()

    # Prepare project ID and path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = args.name.replace(" ", "_")
    project_id = f"{safe_name}_{timestamp}"
    project_root = os.path.abspath(os.path.join("projects", project_id))
    output_dir = os.path.join(project_root, "output")
    logs_dir = os.path.join(project_root, "logs")

    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load and write config template
    template_path = os.path.join("convoke", "config_template.yaml")
    template = load_template(template_path)

    # Insert requirements into template
    if args.requirements:
        req_text = args.requirements
    elif args.requirements_file:
        with open(args.requirements_file, "r", encoding="utf-8") as rf:
            req_text = rf.read()
    else:
        req_text = "TODO: Add project-specific requirements here."

    # Replace both possible placeholders for requirements
    config_content = template.replace(
        "INSERT_PROJECT_REQUIREMENTS_HERE", req_text
    ).replace("TODO: Insert project-specific requirements here", req_text)
    config_path = os.path.join(project_root, "config.yaml")
    write_file(config_path, config_content)

    # Optionally write initial requirements markdown
    initial_md = os.path.join(project_root, "initial_requirements.md")
    write_file(initial_md, req_text)

    print(f"New project initialized: {project_root}")


if __name__ == "__main__":
    main()

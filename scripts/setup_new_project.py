#!/usr/bin/env python3
"""
setup_new_project.py

Helper script to create a new Convoke project directory with config and requirements.
"""
import argparse
import os
import shutil
from datetime import datetime
import yaml
from convoke.agents import (
    create_architect_task,
    create_architect_review_task,
    create_module_manager_task,
    create_module_review_task,
    create_class_manager_task,
    create_class_review_task,
    create_function_manager_task,
    create_function_review_task,
)


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

    # Create project directory structure
    artifacts_dir = os.path.join(project_root, "artifacts")
    outputs_dir = os.path.join(project_root, "outputs")
    logs_dir = os.path.join(project_root, "logs")

    # Create directories
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Get requirements
    if args.requirements:
        req_text = args.requirements
    elif args.requirements_file:
        with open(args.requirements_file, "r", encoding="utf-8") as rf:
            req_text = rf.read()
    else:
        req_text = "TODO: Add project-specific requirements here."

    # Build config dict with agent type placeholders
    config = {
        "project_name": args.name,
        "requirements": req_text,
        "levels": [
            {
                "name": "System Architecture",
                "tasks": [
                    {
                        "name": "System Architecture",
                        "description": "Design the system architecture.",
                        "agent_type": "architect",
                        "review": {
                            "enabled": True,
                            "agent_type": "architect_reviewer",
                        },
                        "refine": {
                            "enabled": True,
                            "cycles": 2,
                        },
                    }
                ],
            },
            {
                "name": "Module Design",
                "tasks": [
                    {
                        "name": "Module Design",
                        "description": "Design the main modules for the system.",
                        "agent_type": "module_manager",
                        "review": {
                            "enabled": True,
                            "agent_type": "module_reviewer",
                        },
                        "refine": {
                            "enabled": True,
                            "cycles": 2,
                        },
                    }
                ],
            },
        ],
    }

    # Write config.yaml
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    # Optionally write initial requirements markdown
    initial_md = os.path.join(project_root, "initial_requirements.md")
    write_file(initial_md, req_text)

    print(f"New project initialized: {project_root}")
    print(
        "\nNOTE: The config.yaml uses agent_type placeholders. When loading the config in your workflow runner, map agent_type to the correct agent/task factory function from convoke.agents."
    )


if __name__ == "__main__":
    main()

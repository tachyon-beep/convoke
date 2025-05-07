import logging
import os
import json
import subprocess


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_api_keys():
    """Ensure required API keys are set for CrewAI agents."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY must be set in the environment.")
    os.environ["OPENAI_API_KEY"] = openai_key  # Explicitly set for downstream consumers
    # Add other keys as needed
    # os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY", "")


def ensure_dir_exists(path: str):
    os.makedirs(path, exist_ok=True)


def write_json_file(path: str, data: dict):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_text_file(path: str, text: str):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def lint_python_code(code: str) -> str:
    """Run flake8 linter on the given Python code string and return the linting output."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp.flush()
        tmp_path = tmp.name
    try:
        result = subprocess.run(["flake8", tmp_path], capture_output=True, text=True)
        lint_output = result.stdout + result.stderr
    finally:
        os.unlink(tmp_path)
    return lint_output.strip()

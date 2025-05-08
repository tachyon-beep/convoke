import logging
import json
import re
from typing import List, Tuple, Optional
from convoke.agents import ItemListOutput


def parse_numbered_list(text: str, max_items: int = 10) -> List[Tuple[str, str]]:
    if not isinstance(text, str):
        return []
    try:
        lines = text.splitlines()
    except Exception:
        return []
    items = []
    current_name = None
    current_desc_lines = []
    for line in lines:
        m = re.match(r"^\s*(\d+|[-*])\.?\s+([^:]+):?\s*(.*)$", line)
        if m:
            if current_name is not None:
                items.append((current_name, "\n".join(current_desc_lines).strip()))
            current_name = m.group(2).strip()
            desc = m.group(3).strip()
            current_desc_lines = [desc] if desc else []
        elif current_name is not None and not re.match(r"^\s*(\d+|[-*])\.?\s+", line):
            current_desc_lines.append(line)
    if current_name is not None:
        items.append((current_name, "\n".join(current_desc_lines).strip()))
    return items[:max_items]


def parse_json_list(
    text: str, max_items: int = 10, logger: Optional[logging.Logger] = None
) -> List[Tuple[str, str]]:
    local_logger = logger or logging.getLogger(__name__)
    if not isinstance(text, str):
        return []
    if not text.strip():
        local_logger.warning(
            "parse_json_list received empty or whitespace-only string."
        )
        return []
    try:
        match = re.search(r"(\[[\s\S]*\])", text)
        if not match:
            local_logger.warning(
                f"parse_json_list: Could not find a JSON array structure in text starting with: {text[:200]}..."
            )
            return []
        json_text = match.group(1)
        data = json.loads(json_text)
        items = []
        if isinstance(data, list):
            for item_dict in data:
                if (
                    isinstance(item_dict, dict)
                    and "name" in item_dict
                    and "description" in item_dict
                ):
                    items.append(
                        (str(item_dict["name"]), str(item_dict["description"]))
                    )
                else:
                    local_logger.warning(
                        f"parse_json_list: Skipping invalid item in JSON list: {item_dict}"
                    )
            return items[:max_items]
        else:
            local_logger.warning(
                f"parse_json_list: Parsed JSON is not a list. Type: {type(data)}. Data: {str(data)[:200]}..."
            )
            return []
    except json.JSONDecodeError as e:
        local_logger.error(
            f"parse_json_list: JSONDecodeError parsing text: {e}. Text snippet: {text[:500]}..."
        )
        return []
    except Exception as e:
        local_logger.error(
            f"parse_json_list: Unexpected error parsing JSON: {e}. Text snippet: {text[:500]}..."
        )
        return []


def extract_items_from_pydantic_output(
    task_output, max_items: int = 10
) -> List[Tuple[str, str]]:
    """
    Extracts items from a Pydantic output model (ItemListOutput).

    Args:
        task_output: The task output containing a Pydantic model.
        max_items: Maximum number of items to extract.

    Returns:
        A list of tuples containing item names and descriptions.
    """
    if hasattr(task_output, "pydantic") and isinstance(
        task_output.pydantic, ItemListOutput
    ):
        return [(item.name, item.description) for item in task_output.pydantic.items][
            :max_items
        ]
    return []

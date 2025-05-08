from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Callable
from convoke.crewai_tools import BaseTool
from convoke.tools import scoped_get_artifact, scoped_save_artifact
from convoke.store import FileSystemArtifactStore
import logging
import re

# Agent decision pattern for continuing iterations
AGENT_CONTINUE_PATTERN = r"@agent\s+Continue:\s*[\"'](.+?)[\"']"


class ItemDetail(BaseModel):
    name: str = Field(description="Name of the item (e.g., module, class, function)")
    description: str = Field(description="Purpose or description of the item")


class ItemListOutput(BaseModel):
    items: List[ItemDetail]


# Initialize the artifact store
artifact_store = FileSystemArtifactStore(
    base_path="./artifacts", logger=logging.getLogger(__name__)
)


# Update tools to use the initialized artifact store
def get_default_tools():
    # Return the tool objects directly without calling them
    return [
        scoped_get_artifact,
        scoped_save_artifact,
    ]


# Check if an agent wants to continue iteration
def check_agent_decision(output: str, default_value: bool = False) -> bool:
    """
    Check if an agent's output contains a decision to continue iterating.
    Format: @agent Continue: "Continue to iterate?"

    Args:
        output: The agent's output text
        default_value: Default value if no decision is found

    Returns:
        True if the agent wants to continue, False otherwise
    """
    match = re.search(AGENT_CONTINUE_PATTERN, output)
    if not match:
        return default_value

    decision_text = match.group(1).strip().lower()

    # Positive indicators
    if any(
        word in decision_text
        for word in ["yes", "continue", "proceed", "true", "iterate"]
    ):
        return True

    # Negative indicators
    if any(
        word in decision_text
        for word in ["no", "stop", "done", "false", "complete", "finished"]
    ):
        return False

    # If unclear, return default
    return default_value


def check_agent_decision(text: str, default_value: bool = True) -> bool:
    """
    Check if the agent has indicated whether to continue or not.
    This looks for the pattern "@agent Continue: " followed by a question or statement.

    Args:
        text: The text to check for a decision
        default_value: The default value to return if no decision is found

    Returns:
        True if the agent wants to continue, False otherwise
    """
    import re

    # Look for the agent decision pattern
    pattern = r"@agent\s+Continue:\s*[\"']([^\"']+)[\"']"
    match = re.search(pattern, text)

    if not match:
        return default_value

    decision_text = match.group(1).strip().lower()

    # Positive responses
    positive_responses = [
        "yes",
        "continue",
        "proceed",
        "go ahead",
        "keep going",
        "iterate",
        "refine",
        "improve",
        "enhance",
        "more",
        "not done",
        "unfinished",
        "incomplete",
    ]

    # Negative responses
    negative_responses = [
        "no",
        "stop",
        "halt",
        "finished",
        "complete",
        "done",
        "satisfied",
        "sufficient",
        "adequate",
        "enough",
        "conclude",
    ]

    # Check for positive indicators
    for phrase in positive_responses:
        if phrase in decision_text:
            return True

    # Check for negative indicators
    for phrase in negative_responses:
        if phrase in decision_text:
            return False

    # If we can't determine a clear answer, use the default
    return default_value


# --- Agent Definitions ---
def create_architect_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Systems Architect",
        goal="Design a robust, modular software system for the given requirements.",
        backstory=(
            "You are a highly experienced systems architect. You break down complex requirements "
            "into high-level modules, ensuring scalability and maintainability.\n"
            "You have access to the GetProjectArtifact and SaveProjectArtifact tools.\n"
            "Use GetProjectArtifact to retrieve any relevant parent or sibling designs.\n"
            "Use SaveProjectArtifact to save your architecture JSON to the correct path."
        ),
        verbose=True,
        allow_delegation=False,
        model="gpt-4o",
        tools=tools,
    )


def create_architect_reviewer_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Architect Reviewer",
        goal="Critique and enhance the system architecture for robustness and clarity.",
        backstory=(
            "You are a peer systems architect, known for your critical eye and ability to spot design flaws. "
            "You review architectural plans and suggest improvements."
        ),
        verbose=True,
        allow_delegation=False,
        tools=tools,
    )


def create_module_manager_agent(tools: Optional[List[BaseTool]] = None):
    tools = tools or get_default_tools()
    return Agent(
        role="Module Manager",
        goal="Design a module, define its classes, and delegate class design to class managers.",
        backstory=(
            "You are responsible for the detailed design of a software module. You identify the "
            "necessary classes and delegate their design to class managers.\n"
            "You have access to the GetProjectArtifact and SaveProjectArtifact tools.\n"
            "Use GetProjectArtifact to retrieve parent architecture or sibling modules if needed.\n"
            "Use SaveProjectArtifact to save your module design JSON to the correct path."
        ),
        verbose=True,
        allow_delegation=True,
        tools=tools or [],
    )


def create_module_reviewer_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Module Reviewer",
        goal="Critique and enhance the module design for cohesion and completeness.",
        backstory=(
            "You are a peer module designer, skilled at reviewing module boundaries and class selection. "
            "You suggest improvements and spot missing elements."
        ),
        verbose=True,
        allow_delegation=False,
        tools=tools,
    )


def create_class_manager_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Class Manager",
        goal="Design a class, define its functions, and delegate function implementation to function managers.",
        backstory=(
            "You are responsible for the design of a class within a module. You identify the necessary "
            "functions/methods and delegate their implementation to function managers.\n"
            "You have access to the GetProjectArtifact and SaveProjectArtifact tools.\n"
            "Use GetProjectArtifact to retrieve parent module or sibling classes if needed.\n"
            "Use SaveProjectArtifact to save your class design JSON to the correct path."
        ),
        verbose=True,
        allow_delegation=True,
        tools=tools,
    )


def create_class_reviewer_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Class Reviewer",
        goal="Critique and enhance the class design for clarity and extensibility.",
        backstory=(
            "You are a peer class designer, skilled at reviewing class responsibilities and method selection. "
            "You suggest improvements and spot missing or redundant methods."
        ),
        verbose=True,
        allow_delegation=False,
        tools=tools,
    )


def create_function_manager_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Function Manager",
        goal="Implement a function or method as specified by the class manager.",
        backstory=(
            "You are responsible for implementing a single function or method, ensuring it meets the "
            "requirements and integrates with the class design."
        ),
        verbose=True,
        allow_delegation=False,
        model="gpt-3.5-turbo",
        tools=tools,
    )


def create_function_reviewer_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Function Reviewer",
        goal="Critique and enhance the function implementation for correctness and style.",
        backstory=(
            "You are a peer function implementer, skilled at reviewing code for bugs, clarity, and efficiency. "
            "You suggest improvements and spot potential issues."
        ),
        verbose=True,
        allow_delegation=False,
        tools=tools,
    )


def create_test_developer_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Test Developer",
        goal="Write a unit test for the given function implementation.",
        backstory="You are responsible for writing clear, effective unit tests for Python functions, using pytest conventions.",
        verbose=True,
        allow_delegation=False,
        model="gpt-3.5-turbo",
        tools=tools,
    )


def create_test_reviewer_agent(tools: Optional[List[BaseTool]] = None):
    return Agent(
        role="Test Reviewer",
        goal="Critique and enhance the unit test for correctness and coverage.",
        backstory="You are a peer test developer, skilled at reviewing unit tests for completeness and effectiveness.",
        verbose=True,
        allow_delegation=False,
        tools=tools,
    )


# --- Task Definitions ---
def create_architect_task(requirements, tools: Optional[List[BaseTool]] = None):
    agent = create_architect_agent(tools)
    return Task(
        description=(
            f"Analyze the following requirements and design a modular system.\n"
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields.\n"
            f"Example: {{'items': [{{'name': 'ExampleModule', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or text outside the JSON object itself.\n"
            f"Finally, use the 'SaveProjectArtifact' tool to save your output JSON to 'system_architecture.json'.\n"
            f"Requirements: {requirements}"
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name' and 'description' fields, strictly conforming to the ItemListOutput schema. No extra text.",
        agent=agent,
        output_pydantic=ItemListOutput,
    )


def create_architect_review_task(arch_task, tools: Optional[List[BaseTool]] = None):
    agent = create_architect_reviewer_agent(tools)
    return Task(
        description=(
            "Review the proposed system architecture. Critique its modularity, scalability, and clarity. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the architecture.",
        agent=agent,
        context=[arch_task],
        human_input=True,
    )


def create_module_manager_task(
    module_name, module_description, tools: Optional[List[BaseTool]] = None
):
    agent = create_module_manager_agent(tools)
    module_filename = f"{module_name.replace(' ', '_')}_module_design.json"
    return Task(
        description=(
            f"You are the Module Manager for module '{module_name}'. High-level description: {module_description}.\n"
            f"Use the 'GetProjectArtifact' tool to retrieve 'system_architecture.json' if needed for context.\n"
            f"Design the classes for this module.\n"
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields. "
            f"Example: {{'items': [{{'name': 'ExampleClass', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself.\n"
            f"Finally, use the 'SaveProjectArtifact' tool to save your output JSON to '{module_filename}'."
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name' and 'description' fields, strictly conforming to the ItemListOutput schema. No extra text.",
        agent=agent,
        output_pydantic=ItemListOutput,
    )


def create_module_review_task(mod_task, tools: Optional[List[BaseTool]] = None):
    agent = create_module_reviewer_agent(tools)
    return Task(
        description=(
            "Review the proposed module design. Critique its cohesion, completeness, and class selection. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the module design.",
        agent=agent,
        context=[mod_task],
    )


def create_class_manager_task(
    module_name, class_name, class_description, tools: Optional[List[BaseTool]] = None
):
    agent = create_class_manager_agent(tools)
    module_filename = f"{module_name.replace(' ', '_')}_module_design.json"
    return Task(
        description=(
            f"You are the Class Manager for class '{class_name}' in module '{module_name}'.\n"
            f"High-level description: {class_description}.\n"
            f"Use the 'GetProjectArtifact' tool to retrieve your parent module's '{module_filename}' if needed for context.\n"
            f"Design the functions/methods for this class.\n"
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name', 'description', and 'signature' fields.\n"
            f"Example: {{'items': [{{'name': 'example_function', 'description': 'Example function.', 'signature': 'def example_function(param1: str) -> bool:'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself.\n"
            f"Finally, use the 'SaveProjectArtifact' tool to save your output JSON to '{class_name.replace(' ', '_')}_class_design.json'."
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name', 'description', and 'signature' fields, strictly conforming to the FunctionListOutput schema. No extra text.",
        agent=agent,
        output_pydantic=FunctionListOutput,
    )


def create_class_review_task(cls_task, tools: Optional[List[BaseTool]] = None):
    agent = create_class_reviewer_agent(tools)
    return Task(
        description=(
            "Review the proposed class design. Critique its clarity, extensibility, and method selection. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the class design.",
        agent=agent,
        context=[cls_task],
    )


def create_function_manager_task(
    function_name,
    function_description,
    class_name="",
    tools: Optional[List[BaseTool]] = None,
):
    agent = create_function_manager_agent(tools)
    class_design_file = (
        f"{class_name.replace(' ', '_')}_class_design.json"
        if class_name
        else "class_design.json"
    )
    return Task(
        description=(
            f"You are the Function Manager for function '{function_name}'. Description: {function_description}.\n"
            f"Use the 'GetProjectArtifact' tool to retrieve your parent class's '{class_design_file}' if needed for context.\n"
            f"Implement the function/method as specified, providing the full code implementation with a docstring.\n"
            f"Finally, use the 'SaveProjectArtifact' tool to save your code to '{function_name}.py'."
        ),
        expected_output="The complete code for the function/method, with a docstring.",
        agent=agent,
    )


def create_function_review_task(fn_task, tools: Optional[List[BaseTool]] = None):
    agent = create_function_reviewer_agent(tools)
    return Task(
        description=(
            "Review the function implementation. Critique its correctness, clarity, and efficiency. "
            "Suggest specific improvements or enhancements."
        ),
        expected_output="A critique and enhancement suggestions for the function implementation.",
        agent=agent,
        context=[fn_task],
    )


def create_test_developer_task(
    function_name, function_code, tools: Optional[List[BaseTool]] = None
):
    agent = create_test_developer_agent(tools)
    return Task(
        description=(
            f"Write a pytest-style unit test for the following function '{function_name}'. "
            f"The test should be in a single code block, with no extra text.\n"
            f"Function implementation:\n{function_code}"
        ),
        expected_output="A complete pytest-style unit test for the function, as a code block.",
        agent=agent,
    )


def create_test_review_task(test_task, tools: Optional[List[BaseTool]] = None):
    agent = create_test_reviewer_agent(tools)
    return Task(
        description="Review the unit test for correctness, coverage, and clarity. Suggest improvements if needed.",
        expected_output="A critique and enhancement suggestions for the unit test.",
        agent=agent,
        context=[test_task],
    )

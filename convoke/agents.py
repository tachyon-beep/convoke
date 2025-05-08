from crewai import Agent, Task
from pydantic import BaseModel, Field
from typing import List, Optional
from convoke.crewai_tools import BaseTool


class ItemDetail(BaseModel):
    name: str = Field(description="Name of the item (e.g., module, class, function)")
    description: str = Field(description="Purpose or description of the item")


class ItemListOutput(BaseModel):
    items: List[ItemDetail]


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
    return Task(
        description=(
            f"You are the Module Manager for module '{module_name}'. High-level description: {module_description}.\n"
            f"Use the 'GetProjectArtifact' tool to retrieve 'system_architecture.json' if needed for context.\n"
            f"Design the classes for this module.\n"
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields. "
            f"Example: {{'items': [{{'name': 'ExampleClass', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself.\n"
            f"Finally, use the 'SaveProjectArtifact' tool to save your output JSON to 'module_design.json'."
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
    class_name, class_description, tools: Optional[List[BaseTool]] = None
):
    agent = create_class_manager_agent(tools)
    return Task(
        description=(
            f"You are the Class Manager for class '{class_name}'. High-level description: {class_description}.\n"
            f"Use the 'GetProjectArtifact' tool to retrieve your parent module's 'module_design.json' if needed for context.\n"
            f"Design the functions/methods for this class.\n"
            f"Output MUST be a JSON object with a single key 'items', whose value is a list of objects, each with 'name' and 'description' fields. "
            f"Example: {{'items': [{{'name': 'ExampleFunction', 'description': 'This is an example.'}}]}}\n"
            f"Do not include any markdown, comments, or any text outside the JSON object itself.\n"
            f"Finally, use the 'SaveProjectArtifact' tool to save your output JSON to 'class_design.json'."
        ),
        expected_output="A JSON object with an 'items' key, whose value is a list of objects with 'name' and 'description' fields, strictly conforming to the ItemListOutput schema. No extra text.",
        agent=agent,
        output_pydantic=ItemListOutput,
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
    function_name, function_description, tools: Optional[List[BaseTool]] = None
):
    agent = create_function_manager_agent(tools)
    return Task(
        description=(
            f"You are the Function Manager for function '{function_name}'. Description: {function_description}.\n"
            f"Use the 'GetProjectArtifact' tool to retrieve your parent class's 'class_design.json' if needed for context.\n"
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

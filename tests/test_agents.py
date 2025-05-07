from convoke.agents import (
    create_architect_agent,
    create_architect_reviewer_agent,
    create_module_manager_agent,
    create_module_reviewer_agent,
    create_class_manager_agent,
    create_class_reviewer_agent,
    create_function_manager_agent,
    create_function_reviewer_agent,
    create_test_developer_agent,
    create_test_reviewer_agent,
    create_architect_task,
    create_architect_review_task,
    create_module_manager_task,
    create_module_review_task,
    create_class_manager_task,
    create_class_review_task,
    create_function_manager_task,
    create_function_review_task,
    create_test_developer_task,
    create_test_review_task,
)


def test_agent_factories():
    assert hasattr(create_architect_agent(), "role")
    assert hasattr(create_architect_reviewer_agent(), "role")
    assert hasattr(create_module_manager_agent(), "role")
    assert hasattr(create_module_reviewer_agent(), "role")
    assert hasattr(create_class_manager_agent(), "role")
    assert hasattr(create_class_reviewer_agent(), "role")
    assert hasattr(create_function_manager_agent(), "role")
    assert hasattr(create_function_reviewer_agent(), "role")
    assert hasattr(create_test_developer_agent(), "role")
    assert hasattr(create_test_reviewer_agent(), "role")


def test_task_factories():
    t = create_architect_task("reqs")
    assert hasattr(t, "description")
    assert hasattr(t, "agent")
    assert hasattr(t, "expected_output")
    t2 = create_architect_review_task(t)
    assert hasattr(t2, "description")
    t3 = create_module_manager_task("mod", "desc")
    assert hasattr(t3, "description")
    t4 = create_module_review_task(t3)
    assert hasattr(t4, "description")
    t5 = create_class_manager_task("cls", "desc")
    assert hasattr(t5, "description")
    t6 = create_class_review_task(t5)
    assert hasattr(t6, "description")
    t7 = create_function_manager_task("fn", "desc")
    assert hasattr(t7, "description")
    t8 = create_function_review_task(t7)
    assert hasattr(t8, "description")
    t9 = create_test_developer_task("fn", "code")
    assert hasattr(t9, "description")
    t10 = create_test_review_task(t9)
    assert hasattr(t10, "description")

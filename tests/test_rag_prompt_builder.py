import pytest

from leoai.rag_prompt_builder import (
    AgentOrchestrator,
    CURRENT_QUESTION_INSTRUCTIONS,
    PLACE_FOLLOWUP_INSTRUCTIONS,
)


@pytest.mark.parametrize(
    "question",
    [
        "How do I write a Python function?",
        "What are the opening hours of the selected church?",
        "2",
        "option 2",
        "chọn 2",
        "history",
        "Why?",
    ],
)
def test_current_request_and_selection_rules_reach_prompt_and_system(question):
    context = {
        "user_context": {
            "selected_place": {"id": 42, "name": "Selected Church"},
            "place_choices": [
                {"id": 42, "name": "Selected Church"},
                {"id": 7, "name": "Local Cafe"},
            ],
        },
        "context_summary": "Previously discussing church history.",
    }

    prompt = AgentOrchestrator().build_prompt(question, context, "English")

    assert CURRENT_QUESTION_INSTRUCTIONS in prompt.prompt_text
    assert CURRENT_QUESTION_INSTRUCTIONS in prompt.system_instruction
    assert PLACE_FOLLOWUP_INSTRUCTIONS in prompt.system_instruction
    assert f"### User’s Current Question\n{question}" in prompt.prompt_text
    assert "Local Cafe" in prompt.prompt_text
    assert prompt.prompt_text.index("### Current request priority") < (
        prompt.prompt_text.index("### User Context")
    )
    assert context["user_context"]["selected_place"]["id"] == 42


@pytest.mark.parametrize(
    ("language", "expected"),
    [("", "the user's language"), ("Vietnamese", "Vietnamese")],
)
def test_language_instruction_is_explicit_with_or_without_target(language, expected):
    prompt = AgentOrchestrator().build_prompt("Why?", {}, language)

    assert f"Respond in {expected}." in prompt.prompt_text
    assert "No place has been selected." in prompt.prompt_text
    assert "End with an insightful remark" not in prompt.prompt_text

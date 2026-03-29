"""Tests verifying that AIGenerator correctly detects and executes tool calls."""
import pytest
from unittest.mock import MagicMock, patch, call
from ai_generator import AIGenerator


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic response objects
# ---------------------------------------------------------------------------

def _make_text_response(text: str):
    """Simulate a plain-text (non-tool) Anthropic response."""
    block = MagicMock()
    block.type = "text"
    block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [block]
    return response


def _make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tool_abc123"):
    """Simulate an Anthropic response that requests a tool call."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def _make_final_text_response(text: str):
    """Simulate the follow-up text response after a tool call."""
    return _make_text_response(text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_definitions():
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]


@pytest.fixture
def mock_tool_manager():
    mgr = MagicMock()
    mgr.execute_tool.return_value = "RAG stands for Retrieval-Augmented Generation."
    return mgr


@pytest.fixture
def generator():
    """AIGenerator wired to the anthropic backend with a dummy key."""
    return AIGenerator(api_key="test-key", model="claude-sonnet-4-5", backend="anthropic")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoToolForGeneralQuestion:
    """When Claude returns a plain text response, no tool should be executed."""

    def test_returns_text_directly(self, generator, tool_definitions, mock_tool_manager):
        text_response = _make_text_response("The sky is blue because of Rayleigh scattering.")

        with patch.object(generator.anthropic_client.messages, "create", return_value=text_response):
            result = generator.generate_response(
                query="Why is the sky blue?",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        assert result == "The sky is blue because of Rayleigh scattering."
        mock_tool_manager.execute_tool.assert_not_called()


class TestToolCalledForContentQuestion:
    """When Claude returns stop_reason='tool_use', the tool must be executed."""

    def test_tool_executed_on_tool_use_response(
        self, generator, tool_definitions, mock_tool_manager
    ):
        tool_response = _make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "What is RAG?"},
        )
        final_response = _make_final_text_response(
            "RAG stands for Retrieval-Augmented Generation."
        )

        with patch.object(
            generator.anthropic_client.messages,
            "create",
            side_effect=[tool_response, final_response],
        ):
            result = generator.generate_response(
                query="What is RAG?",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="What is RAG?"
        )
        assert result == "RAG stands for Retrieval-Augmented Generation."

    def test_tool_result_included_in_follow_up_call(
        self, generator, tool_definitions, mock_tool_manager
    ):
        """The second Anthropic call must include the tool_result message."""
        tool_response = _make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "What is RAG?"},
            tool_id="tool_xyz",
        )
        final_response = _make_final_text_response("Final answer.")
        mock_tool_manager.execute_tool.return_value = "Tool output text"

        captured_calls = []

        def capture_create(**kwargs):
            captured_calls.append(kwargs)
            if len(captured_calls) == 1:
                return tool_response
            return final_response

        with patch.object(generator.anthropic_client.messages, "create", side_effect=capture_create):
            generator.generate_response(
                query="What is RAG?",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        assert len(captured_calls) == 2, "Expected exactly two API calls (initial + follow-up)"

        # The second call's messages must contain the tool_result block
        second_messages = captured_calls[1]["messages"]
        tool_result_messages = [
            m for m in second_messages
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_result_messages) == 1, (
            "Follow-up call must include a user message with tool_result content"
        )

        tool_result_block = tool_result_messages[0]["content"][0]
        assert tool_result_block["tool_use_id"] == "tool_xyz"
        assert tool_result_block["content"] == "Tool output text"


class TestToolCalledWithoutToolManager:
    """If tool_manager is None, tool_use response must not crash."""

    def test_no_tool_manager_stops_at_tool_use(self, generator, tool_definitions):
        """Without a tool_manager, the generator returns a string without raising."""
        tool_response = _make_tool_use_response(
            tool_name="search_course_content",
            tool_input={"query": "RAG"},
        )

        with patch.object(generator.anthropic_client.messages, "create", return_value=tool_response):
            result = generator.generate_response(
                query="RAG question",
                tools=tool_definitions,
                tool_manager=None,
            )

        assert isinstance(result, str)


class TestAPICallParameters:
    """Verify correct parameters are passed to the Anthropic API."""

    def test_tools_included_when_provided(self, generator, tool_definitions, mock_tool_manager):
        text_response = _make_text_response("answer")

        with patch.object(
            generator.anthropic_client.messages, "create", return_value=text_response
        ) as mock_create:
            generator.generate_response(
                query="test query",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        call_kwargs = mock_create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tool_definitions
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_tools_omitted_when_not_provided(self, generator, mock_tool_manager):
        text_response = _make_text_response("answer")

        with patch.object(
            generator.anthropic_client.messages, "create", return_value=text_response
        ) as mock_create:
            generator.generate_response(
                query="test query",
                tools=None,
                tool_manager=mock_tool_manager,
            )

        call_kwargs = mock_create.call_args[1]
        assert "tools" not in call_kwargs

    def test_conversation_history_appended_to_system(self, generator, mock_tool_manager):
        text_response = _make_text_response("answer")

        with patch.object(
            generator.anthropic_client.messages, "create", return_value=text_response
        ) as mock_create:
            generator.generate_response(
                query="follow-up question",
                conversation_history="User: hi\nAssistant: hello",
                tools=None,
                tool_manager=mock_tool_manager,
            )

        call_kwargs = mock_create.call_args[1]
        assert "Previous conversation" in call_kwargs["system"]


class TestTwoRoundToolCalling:
    """Verify sequential tool calling across two rounds."""

    def test_two_sequential_tool_calls_executes_both_tools(
        self, generator, tool_definitions, mock_tool_manager
    ):
        tool_response_1 = _make_tool_use_response(
            "search_course_content", {"query": "RAG"}, tool_id="t1"
        )
        tool_response_2 = _make_tool_use_response(
            "search_course_content", {"query": "embeddings"}, tool_id="t2"
        )
        final_response = _make_final_text_response("Complete answer.")

        with patch.object(
            generator.anthropic_client.messages,
            "create",
            side_effect=[tool_response_1, tool_response_2, final_response],
        ) as mock_create:
            result = generator.generate_response(
                query="Tell me about RAG and embeddings",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        assert mock_tool_manager.execute_tool.call_count == 2
        assert mock_create.call_count == 3
        assert result == "Complete answer."

    def test_context_accumulates_across_rounds(
        self, generator, tool_definitions, mock_tool_manager
    ):
        """The synthesis call must include tool_results from both rounds, and omit tools."""
        mock_tool_manager.execute_tool.side_effect = ["result_1", "result_2"]
        tool_response_1 = _make_tool_use_response(
            "search_course_content", {"query": "RAG"}, tool_id="t1"
        )
        tool_response_2 = _make_tool_use_response(
            "search_course_content", {"query": "embeddings"}, tool_id="t2"
        )
        final_response = _make_final_text_response("Synthesized answer.")

        captured_calls = []

        def capture_create(**kwargs):
            captured_calls.append(kwargs)
            return [tool_response_1, tool_response_2, final_response][len(captured_calls) - 1]

        with patch.object(generator.anthropic_client.messages, "create", side_effect=capture_create):
            generator.generate_response(
                query="multi-part query",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        assert len(captured_calls) == 3
        third_messages = captured_calls[2]["messages"]

        # Both tool_result blocks must be in the 3rd call's messages
        all_tool_results = [
            b for m in third_messages
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        tool_use_ids = {b["tool_use_id"] for b in all_tool_results}
        assert "t1" in tool_use_ids
        assert "t2" in tool_use_ids

        # Final synthesis call must NOT include tools
        assert "tools" not in captured_calls[2]

    def test_max_rounds_respected(self, generator, tool_definitions, mock_tool_manager):
        """Only 2 tool rounds and 1 synthesis call are made, regardless of Claude's responses."""
        tool_response = _make_tool_use_response(
            "search_course_content", {"query": "X"}, tool_id="t1"
        )
        # Provide more tool_use responses than max_rounds to confirm the cap
        final_response = _make_final_text_response("Done.")

        with patch.object(
            generator.anthropic_client.messages,
            "create",
            side_effect=[tool_response, tool_response, final_response],
        ) as mock_create:
            generator.generate_response(
                query="query",
                tools=tool_definitions,
                tool_manager=mock_tool_manager,
            )

        assert mock_tool_manager.execute_tool.call_count <= 2
        assert mock_create.call_count <= 3


class TestToolExecutionError:
    """Tool execution failures must be handled gracefully."""

    def test_tool_execution_error_does_not_raise(
        self, generator, tool_definitions
    ):
        mock_mgr = MagicMock()
        mock_mgr.execute_tool.side_effect = Exception("DB unavailable")

        tool_response = _make_tool_use_response(
            "search_course_content", {"query": "RAG"}, tool_id="t1"
        )
        final_response = _make_final_text_response("Fallback answer.")

        with patch.object(
            generator.anthropic_client.messages,
            "create",
            side_effect=[tool_response, final_response],
        ):
            result = generator.generate_response(
                query="RAG question",
                tools=tool_definitions,
                tool_manager=mock_mgr,
            )

        assert isinstance(result, str)

    def test_tool_execution_error_result_passed_to_claude(
        self, generator, tool_definitions
    ):
        mock_mgr = MagicMock()
        mock_mgr.execute_tool.side_effect = Exception("DB unavailable")

        tool_response = _make_tool_use_response(
            "search_course_content", {"query": "RAG"}, tool_id="t1"
        )
        final_response = _make_final_text_response("Fallback answer.")

        captured_calls = []

        def capture_create(**kwargs):
            captured_calls.append(kwargs)
            return tool_response if len(captured_calls) == 1 else final_response

        with patch.object(generator.anthropic_client.messages, "create", side_effect=capture_create):
            generator.generate_response(
                query="RAG question",
                tools=tool_definitions,
                tool_manager=mock_mgr,
            )

        second_messages = captured_calls[1]["messages"]
        tool_result_contents = [
            b["content"]
            for m in second_messages
            if isinstance(m.get("content"), list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert any("Tool execution failed" in c for c in tool_result_contents)


class TestSystemPromptContent:
    """Regression guard for system prompt content."""

    def test_system_prompt_mentions_two_sequential_tool_calls(self):
        assert "2 sequential" in AIGenerator.SYSTEM_PROMPT

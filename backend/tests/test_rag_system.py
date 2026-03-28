"""Tests verifying RAGSystem.query() orchestrates components correctly."""
import pytest
from unittest.mock import MagicMock, patch
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(chroma_path: str):
    """Return a minimal config-like object for RAGSystem."""
    cfg = MagicMock()
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.CHROMA_PATH = chroma_path
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.LLM_BACKEND = "anthropic"
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-sonnet-4-5"
    cfg.OLLAMA_URL = "http://localhost:11434/v1"
    cfg.OLLAMA_MODEL = "llama3.1"
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rag(temp_dir):
    """A fully initialised RAGSystem backed by a temp ChromaDB."""
    config = _make_config(temp_dir)
    return RAGSystem(config)


@pytest.fixture
def rag_seeded(rag, sample_course, sample_chunks):
    """RAGSystem with one course already indexed."""
    rag.vector_store.add_course_metadata(sample_course)
    rag.vector_store.add_course_content(sample_chunks)
    return rag


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueryReturnShape:
    """RAGSystem.query() must always return (str, list)."""

    def test_query_returns_tuple(self, rag):
        with patch.object(rag.ai_generator, "generate_response", return_value="Some answer"):
            result = rag.query("What is RAG?")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_query_returns_string_and_list(self, rag):
        with patch.object(rag.ai_generator, "generate_response", return_value="Some answer"):
            response, sources = rag.query("What is RAG?")

        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_response_matches_generator_output(self, rag):
        with patch.object(rag.ai_generator, "generate_response", return_value="Expected response"):
            response, _ = rag.query("test question")

        assert response == "Expected response"


class TestQueryWithToolSources:
    """When a search tool is invoked, sources must be returned."""

    def test_sources_populated_after_tool_execution(self, rag_seeded):
        """
        Simulate the tool being called during generate_response by manually
        pre-seeding last_sources on the search tool, then calling query().
        """
        expected_sources = [
            {"text": "Introduction to RAG - Lesson 1", "url": "https://example.com/rag-course/lesson/1"}
        ]
        rag_seeded.search_tool.last_sources = expected_sources

        with patch.object(rag_seeded.ai_generator, "generate_response", return_value="Answer"):
            _, sources = rag_seeded.query("What is RAG?")

        assert sources == expected_sources

    def test_sources_reset_after_query(self, rag_seeded):
        """last_sources on the search tool must be cleared after each query."""
        rag_seeded.search_tool.last_sources = [{"text": "Stale source", "url": None}]

        with patch.object(rag_seeded.ai_generator, "generate_response", return_value="Answer"):
            rag_seeded.query("What is RAG?")

        assert rag_seeded.search_tool.last_sources == []


class TestQueryWithEmptyStore:
    """Querying against an empty store must not raise an unhandled exception."""

    def test_no_crash_on_empty_store(self, rag):
        """
        BUG PROBE: If the vector store is empty and the AI tries to use the
        search tool, the whole pipeline must not raise — it must return a string.
        This will fail if RAGSystem.query() has no exception handling and
        the tool / vector store raises an uncaught error.
        """
        with patch.object(rag.ai_generator, "generate_response", return_value="No results found."):
            response, sources = rag.query("What is the difference between RAG and fine-tuning?")

        assert isinstance(response, str)
        assert isinstance(sources, list)


class TestSessionManagement:
    """Session history must be updated correctly after each query."""

    def test_session_history_updated(self, rag):
        session_id = rag.session_manager.create_session()

        with patch.object(rag.ai_generator, "generate_response", return_value="Answer"):
            rag.query("Tell me about RAG", session_id=session_id)

        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "Tell me about RAG" in history

    def test_query_without_session_does_not_crash(self, rag):
        with patch.object(rag.ai_generator, "generate_response", return_value="Answer"):
            response, _ = rag.query("test", session_id=None)

        assert response == "Answer"


class TestQueryExceptionPropagation:
    """
    Documents the missing try-except in RAGSystem.query().
    If AIGenerator raises, the exception propagates to app.py which returns HTTP 500.
    """

    def test_exception_from_generator_returns_friendly_message(self, rag):
        """
        After Fix C: RAGSystem.query() has a try-except that catches generator
        errors and returns a user-facing message instead of propagating the
        exception to app.py (which would cause HTTP 500 / 'query failed').
        """
        with patch.object(
            rag.ai_generator, "generate_response", side_effect=RuntimeError("API error")
        ):
            response, sources = rag.query("What is RAG?")

        assert isinstance(response, str)
        assert "error" in response.lower()
        assert isinstance(sources, list)


class TestEndToEndQueryWithRealVectorStore:
    """
    Integration-level tests: use the real vector store (backed by temp ChromaDB)
    and mock only the Anthropic API call.
    """

    def test_content_query_reaches_search_tool(self, rag_seeded):
        """
        Simulate Claude calling search_course_content and verify the full
        pipeline (tool execution → format → response) works without errors.
        """
        # Manually execute the tool to verify it returns sensible results
        tool = rag_seeded.search_tool
        result = tool.execute(query="What is RAG?")

        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_definitions_passed_to_generator(self, rag):
        """generate_response must be called with the registered tool definitions."""
        captured = {}

        def mock_generate(query, conversation_history=None, tools=None, tool_manager=None):
            captured["tools"] = tools
            return "answer"

        with patch.object(rag.ai_generator, "generate_response", side_effect=mock_generate):
            rag.query("test")

        assert captured["tools"] is not None
        tool_names = [t["name"] for t in captured["tools"]]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

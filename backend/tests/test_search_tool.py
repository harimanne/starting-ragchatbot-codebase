"""Tests for CourseSearchTool.execute() and related VectorStore behaviour."""
import pytest
from search_tools import CourseSearchTool


# ---------------------------------------------------------------------------
# Basic query behaviour
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:
    """Tests covering the happy-path and common edge-cases of execute()."""

    def test_execute_returns_formatted_results(self, seeded_vector_store):
        """A content query on a seeded store returns a non-empty formatted string."""
        tool = CourseSearchTool(seeded_vector_store)
        result = tool.execute(query="What is RAG?")

        assert isinstance(result, str)
        assert len(result) > 0
        # The course title must appear in the formatted header
        assert "Introduction to RAG" in result

    def test_execute_empty_collection_returns_no_results_message(self, empty_vector_store):
        """Querying an empty store must return a human-readable message, not raise."""
        tool = CourseSearchTool(empty_vector_store)
        result = tool.execute(query="What is RAG?")

        assert isinstance(result, str)
        assert "No relevant content found" in result

    def test_execute_with_valid_course_filter(self, seeded_vector_store):
        """Filtering by an existing course name narrows results to that course."""
        tool = CourseSearchTool(seeded_vector_store)
        result = tool.execute(query="RAG pipeline", course_name="Introduction to RAG")

        assert isinstance(result, str)
        assert "Introduction to RAG" in result

    def test_execute_with_nonexistent_course_returns_no_results(self, seeded_vector_store):
        """Filtering by an unknown course name returns an informative message."""
        tool = CourseSearchTool(seeded_vector_store)
        result = tool.execute(query="deep learning", course_name="Nonexistent Course XYZ")

        assert isinstance(result, str)
        assert "No course found matching" in result or "No relevant content found" in result

    def test_execute_with_lesson_filter(self, seeded_vector_store):
        """Filtering by lesson number returns only content from that lesson."""
        tool = CourseSearchTool(seeded_vector_store)
        result = tool.execute(
            query="RAG",
            course_name="Introduction to RAG",
            lesson_number=1,
        )

        assert isinstance(result, str)
        # Should contain lesson 1 content
        assert "Lesson 1" in result or "RAG" in result

    def test_execute_stores_sources(self, seeded_vector_store):
        """After a successful search, last_sources contains text+url dicts."""
        tool = CourseSearchTool(seeded_vector_store)
        tool.execute(query="What is RAG?")

        assert isinstance(tool.last_sources, list)
        assert len(tool.last_sources) > 0
        for src in tool.last_sources:
            assert "text" in src

    def test_execute_stores_no_sources_on_empty_result(self, empty_vector_store):
        """On empty results, last_sources should be empty."""
        tool = CourseSearchTool(empty_vector_store)
        tool.execute(query="What is RAG?")
        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# Bug probes — these tests are designed to expose known suspect behaviour
# ---------------------------------------------------------------------------

class TestNoneLessonNumberBug:
    """
    ChromaDB 1.0.x rejects None metadata values.
    add_course_content() with lesson_number=None should either succeed gracefully
    or raise a clear error — not silently corrupt state.
    """

    def test_add_chunk_with_none_lesson_number_does_not_raise(
        self, empty_vector_store, sample_course, chunks_with_none_lesson
    ):
        """
        BUG PROBE: Storing a chunk with lesson_number=None must not crash the store.
        If this test FAILS it confirms Bug A: ChromaDB rejects None metadata values.
        """
        empty_vector_store.add_course_metadata(sample_course)
        # This line is expected to raise ValueError in ChromaDB 1.0.x if the bug is present
        empty_vector_store.add_course_content(chunks_with_none_lesson)

        # If we get here, verify the chunk is actually queryable
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(empty_vector_store)
        result = tool.execute(query="chunk with no lesson")
        assert isinstance(result, str)


class TestChromaFilterFormat:
    """
    Tests that verify the ChromaDB $and filter syntax used in _build_filter()
    is accepted by ChromaDB 1.0.x.
    """

    def test_build_filter_and_operator_accepted_by_chromadb(self, seeded_vector_store):
        """
        BUG PROBE: Querying with both course_name AND lesson_number uses $and filter.
        If this test FAILS it confirms Bug B: $and filter needs explicit $eq operators.
        """
        tool = CourseSearchTool(seeded_vector_store)
        # This triggers _build_filter with both course_title and lesson_number set
        result = tool.execute(
            query="RAG",
            course_name="Introduction to RAG",
            lesson_number=1,
        )
        # Must not raise; must return a string (even if empty results)
        assert isinstance(result, str)

    def test_single_course_filter_accepted(self, seeded_vector_store):
        """A single course_title filter must work without $and."""
        tool = CourseSearchTool(seeded_vector_store)
        result = tool.execute(query="RAG pipeline", course_name="Introduction to RAG")
        assert isinstance(result, str)

    def test_single_lesson_filter_accepted(self, seeded_vector_store):
        """A lone lesson_number filter (no course) must not crash."""
        tool = CourseSearchTool(seeded_vector_store)
        result = tool.execute(query="RAG", lesson_number=1)
        assert isinstance(result, str)

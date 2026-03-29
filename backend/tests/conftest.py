"""Shared fixtures for all tests."""

import sys
import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch

# Add backend directory to sys.path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore


@pytest.fixture
def sample_course():
    """A well-formed Course with two lessons."""
    return Course(
        title="Introduction to RAG",
        course_link="https://example.com/rag-course",
        instructor="Jane Smith",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is RAG?",
                lesson_link="https://example.com/rag-course/lesson/1",
            ),
            Lesson(
                lesson_number=2,
                title="Building Pipelines",
                lesson_link="https://example.com/rag-course/lesson/2",
            ),
        ],
    )


@pytest.fixture
def sample_chunks(sample_course):
    """Valid CourseChunk list (no None lesson_numbers) for seeding."""
    return [
        CourseChunk(
            content="RAG stands for Retrieval-Augmented Generation. It combines search with LLMs.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Building a RAG pipeline involves a vector store and an embedding model.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def chunks_with_none_lesson(sample_course):
    """CourseChunk list that includes a chunk with lesson_number=None (triggers ChromaDB bug)."""
    return [
        CourseChunk(
            content="This chunk has no lesson number.",
            course_title=sample_course.title,
            lesson_number=None,  # This is the problematic value
            chunk_index=0,
        ),
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for ChromaDB; cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def empty_vector_store(temp_dir):
    """A VectorStore backed by a fresh (empty) ChromaDB instance."""
    return VectorStore(
        chroma_path=temp_dir,
        embedding_model="all-MiniLM-L6-v2",
        max_results=5,
    )


@pytest.fixture
def seeded_vector_store(empty_vector_store, sample_course, sample_chunks):
    """A VectorStore that already contains one course and its chunks."""
    empty_vector_store.add_course_metadata(sample_course)
    empty_vector_store.add_course_content(sample_chunks)
    return empty_vector_store

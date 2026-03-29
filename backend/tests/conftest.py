"""Shared fixtures for all tests."""

import sys
import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional

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


# ---------------------------------------------------------------------------
# API testing fixtures
# ---------------------------------------------------------------------------

class _QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class _SourceLink(BaseModel):
    text: str
    url: Optional[str] = None

class _QueryResponse(BaseModel):
    answer: str
    sources: List[_SourceLink]
    session_id: str

class _CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def _build_test_app(rag_system) -> FastAPI:
    """Build a FastAPI app with the same API routes as app.py but no static files mount."""
    app = FastAPI(title="Test RAG API")

    @app.post("/api/query", response_model=_QueryResponse)
    async def query_documents(request: _QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return _QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        rag_system.session_manager.clear_session(session_id)
        return {"status": "cleared"}

    @app.get("/api/courses", response_model=_CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return _CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def mock_rag_system():
    """A MagicMock RAGSystem pre-configured with sensible return values."""
    mock = MagicMock()
    mock.session_manager.create_session.return_value = "test-session-abc"
    mock.query.return_value = ("Test answer about RAG.", [])
    mock.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to RAG", "Building Pipelines"],
    }
    return mock


@pytest.fixture
def api_client(mock_rag_system):
    """A TestClient for the test FastAPI app, backed by a mocked RAGSystem."""
    app = _build_test_app(mock_rag_system)
    with TestClient(app) as client:
        yield client

"""Tests for FastAPI API endpoints (/api/query, /api/courses, /api/session)."""
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    """Tests for POST /api/query."""

    def test_returns_200_with_answer(self, api_client):
        resp = api_client.post("/api/query", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Test answer about RAG."
        assert isinstance(body["sources"], list)
        assert isinstance(body["session_id"], str)

    def test_auto_creates_session_when_none_provided(self, api_client):
        resp = api_client.post("/api/query", json={"query": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test-session-abc"

    def test_uses_provided_session_id(self, api_client, mock_rag_system):
        resp = api_client.post(
            "/api/query",
            json={"query": "Hello", "session_id": "existing-session"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "existing-session"
        # create_session must NOT be called when a session_id is supplied
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_returns_500_when_rag_raises(self, api_client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("vector store unavailable")
        resp = api_client.post("/api/query", json={"query": "crash"})
        assert resp.status_code == 500
        assert "vector store unavailable" in resp.json()["detail"]

    def test_missing_query_field_returns_422(self, api_client):
        resp = api_client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_sources_returned_from_rag(self, api_client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Found it.",
            [{"text": "Lesson 1", "url": "https://example.com/lesson/1"}],
        )
        resp = api_client.post("/api/query", json={"query": "lesson details"})
        assert resp.status_code == 200
        sources = resp.json()["sources"]
        assert len(sources) == 1
        assert sources[0]["text"] == "Lesson 1"
        assert sources[0]["url"] == "https://example.com/lesson/1"

    def test_sources_without_url_are_accepted(self, api_client, mock_rag_system):
        mock_rag_system.query.return_value = (
            "Found it.",
            [{"text": "Some source", "url": None}],
        )
        resp = api_client.post("/api/query", json={"query": "anything"})
        assert resp.status_code == 200
        assert resp.json()["sources"][0]["url"] is None


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:
    """Tests for GET /api/courses."""

    def test_returns_200_with_stats(self, api_client):
        resp = api_client.get("/api/courses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 2
        assert "Introduction to RAG" in body["course_titles"]
        assert "Building Pipelines" in body["course_titles"]

    def test_returns_500_when_analytics_raises(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = Exception("db error")
        resp = api_client.get("/api/courses")
        assert resp.status_code == 500
        assert "db error" in resp.json()["detail"]

    def test_empty_catalog_returns_zero_courses(self, api_client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = api_client.get("/api/courses")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestSessionEndpoint:
    """Tests for DELETE /api/session/{session_id}."""

    def test_returns_200_and_cleared_status(self, api_client):
        resp = api_client.delete("/api/session/my-session")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared"}

    def test_delegates_to_session_manager(self, api_client, mock_rag_system):
        api_client.delete("/api/session/abc-123")
        mock_rag_system.session_manager.clear_session.assert_called_once_with("abc-123")

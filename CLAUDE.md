# CLAUDE.md

## Package Management
- Always use `uv` for installing packages and running Python commands
- Never use `pip` directly

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Install dependencies (from repo root)
uv sync

# Set up environment
cp .env.example .env  # then add your ANTHROPIC_API_KEY

# Start the server (from repo root)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App runs at `http://localhost:8000`, API docs at `http://localhost:8000/docs`.

The server auto-loads all `.txt` files from `docs/` into ChromaDB on startup. ChromaDB persists to `backend/chroma_db/` — existing courses are skipped on restart (no re-indexing unless `clear_existing=True`).

## Architecture

**Single-process, no separate frontend server.** FastAPI serves both the REST API and the `frontend/` directory as static files from port 8000.

### RAG Query Flow
1. `POST /api/query` → `RAGSystem.query()` → `AIGenerator` sends messages + tools to Claude
2. Claude invokes `CourseSearchTool` → `VectorStore.search()` queries ChromaDB
3. ChromaDB returns top-k chunks → Claude generates answer → response returned with sources

### Document Ingestion Flow
Triggered at startup via `app.py:startup_event()` and on `POST /api/upload` (if implemented):
`docs/*.txt` → `DocumentProcessor.process_course_document()` → `(Course, List[CourseChunk])` → `VectorStore.add_course_metadata()` + `VectorStore.add_course_content()`

### Key Design Decisions
- **Two ChromaDB collections:** `course_catalog` (course-level metadata for fuzzy name resolution) and `course_content` (chunk text for semantic search). Course name resolution always goes through `catalog` first before querying `content`.
- **Tool-calling architecture:** Claude is given `CourseSearchTool` and decides when to invoke it. The tool definition lives in `search_tools.py`; execution is handled by `AIGenerator` which detects `tool_use` blocks in Claude's response and loops until a final text response is produced.
- **Session memory:** `SessionManager` keeps the last `MAX_HISTORY=2` exchanges per session (in-memory dict, not persisted). Sessions are keyed by UUID passed from the frontend.
- **All config in one place:** `backend/config.py` — model name, chunk size/overlap, max results, ChromaDB path, and conversation history limit are all set here.

### Document Format
Course `.txt` files must follow this structure for correct parsing:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<lesson content>

Lesson 1: <title>
...
```
`DocumentProcessor` falls back to treating the whole file as one document if no `Lesson N:` markers are found.

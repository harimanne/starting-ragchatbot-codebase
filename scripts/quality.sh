#!/bin/bash
# Development quality checks: formatting and tests

set -e

cd "$(dirname "$0")/.."

echo "=== Formatting check (black) ==="
uv run black --check backend/ main.py
echo "Formatting: OK"

echo ""
echo "=== Tests (pytest) ==="
cd backend && uv run pytest tests/ -v

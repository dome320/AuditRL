.PHONY: install dev backend frontend test lint audit eval-p1

install:
	cd backend && uv sync
	cd frontend && npm install

dev:
	$(MAKE) -j2 backend frontend

backend:
	cd backend && uv run uvicorn auditrl.api.app:app --reload --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest -q

lint:
	cd backend && uv run ruff check . && uv run ruff format --check .

# Usage: make audit REPORT=path/to/report.md SOURCES=path/to/sources_dir
audit:
	cd backend && uv run auditrl audit $(abspath $(REPORT)) --sources $(abspath $(SOURCES))

eval-p1:
	cd backend && uv run auditrl eval-p1 ../data/curated

.PHONY: install dev backend frontend test lint audit eval-p1 \
	bench-fetch bench-generate bench-freeze bench-eval bench-compare

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

# Phase 2 benchmark (see README "Benchmark"). MODEL / SPLIT are optional.
bench-fetch:
	cd backend && uv run auditrl bench fetch

bench-generate:
	cd backend && uv run auditrl bench generate $(if $(MODEL),--model $(MODEL))

bench-freeze:
	cd backend && uv run auditrl bench freeze

bench-eval:
	cd backend && uv run auditrl bench eval $(if $(MODEL),--model $(MODEL)) --split $(or $(SPLIT),test)

bench-compare:
	cd backend && uv run auditrl bench compare

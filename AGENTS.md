# AGENTS.md

## Project Overview

Triton MCP is a Model Context Protocol (MCP) server that indexes NVIDIA Triton Inference Server documentation and provides tools for searching, configuring, deploying, and optimizing Triton applications.

## Project Structure

```
src/triton_mcp/
├── config.py          # Constants: base URLs, backend info, GitHub sources, topics
├── crawler.py          # Web crawler (docs site) + GitHub source crawler
├── indexer.py          # Document chunking (tiktoken) + ChromaDB + SQLite FTS5
├── search.py           # Semantic/keyword/hybrid search engine (RRF)
├── prompts.py          # System prompts: developer, troubleshooter, optimizer
├── server.py           # FastMCP server: 11 tools, 2 resources, 3 prompts
└── scripts/
    └── index_docs.py   # CLI: triton-index (crawl + index everything)
```

## Key Commands

```bash
# Install (editable, with dev deps including ruff)
uv sync --group dev

# Rebuild the search index from scratch (docs + GitHub sources)
triton-index                          # full: docs + GitHub
triton-index --skip-github            # docs only (faster)
triton-index --skip-docs              # GitHub sources only

# Start MCP server (streamable HTTP on port 8080)
triton-mcp
TRITON_MCP_PORT=9090 triton-mcp      # custom port

# Run linter and formatter (MUST run before committing)
uv run ruff check src/ tests/         # lint check
uv run ruff format src/ tests/        # auto-format

# Run tests
pytest tests/ -v                       # all tests (integration + unit + direct)
pytest tests/ -v -k "TestDirect"       # direct function tests only (no server start)
pytest tests/ -v -k "TestUnit"         # unit tests only (no server, no index needed)
pytest tests/ -v -k "TestMCP"          # MCP client-server tests only
```

## MCP Server Details

### Transport: Streamable HTTP

Default `http://0.0.0.0:8080/mcp`. Configurable via `TRITON_MCP_PORT` and `TRITON_MCP_HOST` env vars.

### Data Sources Indexed

| Source | Type | ~Pages |
|--------|------|--------|
| docs.nvidia.com Triton docs | HTML crawl | ~194 |
| triton-inference-server/server | GitHub (MD) | ~13 |
| triton-inference-server/client | GitHub (MD + Python) | ~25 |
| triton-inference-server/perf_analyzer | GitHub (MD) | ~8 |
| triton-inference-server/model_analyzer | GitHub (MD) | ~9 |

Index stored at `~/.triton_mcp_index/` (ChromaDB + SQLite). Re-run `triton-index` to update.

### Tools (11)

| Tool | Purpose |
|------|---------|
| `search_docs(query, mode, k)` | Search docs (semantic/keyword/hybrid) |
| `get_page(url)` | Get full page content by URL |
| `list_backends()` | List 9 Triton backends with descriptions |
| `get_model_config_template(backend, ...)` | Generate config.pbtxt templates |
| `get_deployment_guide(topic)` | Docker/K8s/model_repo/perf guides |
| `best_practices(topic)` | Best practices per topic (8 topics) |
| `list_doc_pages()` | List all 256 indexed pages |
| `analyze_config(config_pbtxt)` | **Analyze config.pbtxt, flag issues, suggest optimizations** |
| `python_client_help(task)` | **Search tritonclient source & examples** |
| `perf_test_guide(model_name, ...)` | **Generate perf_analyzer/model_analyzer commands** |
| `model_optimization_guide(backend, ...)` | **Backend-specific optimization guide** |

### Resources

- `triton://docs/index` — JSON index of all pages
- `triton://backends` — JSON list of backends

### Prompts

- `developer(task)` — System prompt for building Triton apps
- `troubleshooter(error_message, model_name?, backend?)` — Debug deployment issues
- `optimizer(model_name, backend, current_config?)` — Performance optimization

## Architecture Decisions

- **Embeddings**: ChromaDB built-in `all-MiniLM-L6-v2` (no external API needed)
- **Search**: Hybrid = Reciprocal Rank Fusion of semantic (cosine) + keyword (BM25/FTS5)
- **Chunking**: 512 tokens with 64-token overlap via tiktoken (cl100k_base)
- **No external LLM dependency**: All tools are deterministic search/retrieval/generation
- **Global state**: Search engine is a module-level singleton with lazy DB connections
- **Server lifecycle**: FastMCP lifespan opens/closes DB connections; `_get_engine()` reconnects if closed

## Code Conventions

- Type hints on all function signatures
- `from __future__ import annotations` at module top
- Docstrings on all public functions (used by MCP tool descriptions)
- No comments unless explicitly requested
- `config.py` is the single source of constants — no magic strings elsewhere
- Tests use Arrange-Act-Assert pattern with clear section comments
- Always run `ruff check` and `ruff format` before committing

## Common Tasks

### Adding a new tool

1. Add tool function in `server.py` with `@mcp.tool()` decorator and docstring
2. Add tool name to `EXPECTED_TOOLS` in `tests/test_integration.py`
3. Add a direct-function test in `TestDirectFunctionCalls`
4. Optionally add an MCP integration test in `TestMCPToolExecution`
5. Update README tools table

### Adding a new GitHub source

1. Add repo config to `GITHUB_SOURCES` in `config.py` with `repo`, `branch`, `paths`, and optional `extra_glob_patterns`
2. Re-run `triton-index` to pick up the new source

### Changing the index schema

1. Edit `indexer.py` (chunking) or `search.py` (search logic)
2. Delete `~/.triton_mcp_index/` to force a rebuild
3. Run `triton-index`
4. Run tests

## Testing

Integration tests use a real MCP server (started on port 9876 via subprocess). Direct function tests don't need a server. Unit tests don't need a server or an index. All other tests require the index to be built (`triton-index` must have been run at least once).

```bash
pytest tests/ -v                             # everything
pytest tests/ -v -k "TestDirect"             # fast, no server needed
pytest tests/ -v -k "TestUnit"               # unit tests, no server or index needed
pytest tests/ -v -k "TestMCPToolDiscovery"   # needs server, tests tool listing
```
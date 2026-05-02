# Triton MCP

MCP server for NVIDIA Triton Inference Server documentation. Crawls and indexes the official docs, providing semantic search (via ChromaDB embeddings), keyword search (via SQLite FTS5), and hybrid search (Reciprocal Rank Fusion), plus tools for generating model configurations and deployment guidance.

## Quick Start

```bash
# Install
pip install -e .

# Index documentation + GitHub sources (first time, takes ~2-3 minutes)
triton-index

# Index only docs (skip GitHub sources)
triton-index --skip-github

# Index only GitHub sources (skip docs site)
triton-index --skip-docs

# Start the MCP server (Streamable HTTP on port 8080)
triton-mcp
```

The index is stored at `~/.triton_mcp_index/` and persists across restarts. Re-run `triton-index` to update.

**Data sources indexed:**
- Official docs site (~200 pages)
- GitHub: `triton-inference-server/server` — model config, backends, protocol
- GitHub: `triton-inference-server/client` — Python client source & examples
- GitHub: `triton-inference-server/perf_analyzer` — perf_analyzer & genai-perf docs
- GitHub: `triton-inference-server/model_analyzer` — model profile/optimization docs

## Tools

| Tool | Description |
|------|-------------|
| `search_docs(query, mode, k)` | Search Triton docs — `semantic`, `keyword`, or `hybrid` (default) |
| `get_page(url)` | Get full content of a documentation page |
| `list_backends()` | List all Triton backends with descriptions |
| `get_model_config_template(backend, ...)` | Generate `config.pbtxt` templates |
| `get_deployment_guide(topic)` | Docker, K8s, model repo, performance guides |
| `best_practices(topic)` | Best practices by topic |
| `list_doc_pages()` | List all indexed documentation pages |
| `analyze_config(config_pbtxt)` | **Analyze a config.pbtxt and suggest optimizations** |
| `python_client_help(task)` | **Python client (tritonclient) help with code examples** |
| `perf_test_guide(model_name, ...)` | **Generate perf_analyzer/model_analyzer commands** |
| `model_optimization_guide(backend, ...)` | **Backend-specific optimization recommendations** |

## Resources

| URI | Content |
|-----|---------|
| `triton://docs/index` | JSON index of all doc pages |
| `triton://backends` | JSON list of available backends |

## Prompts

| Name | Description |
|------|-------------|
| `developer` | System prompt for building Triton apps |
| `troubleshooter` | Prompt for debugging deployment issues |
| `optimizer` | Prompt for optimizing model performance |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `TRITON_MCP_PORT` | `8080` | HTTP port for streamable-http transport |
| `TRITON_MCP_HOST` | `0.0.0.0` | Bind host |

## Adding to MCP Clients

### OpenCode

Add to your `opencode.json` config:

```json
{
  "mcp": {
    "triton": {
      "type": "remote",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

For a remote server, replace `localhost` with your server address.

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "triton": {
      "type": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### Claude Code (CLI)

```bash
claude mcp add --transport http triton http://localhost:8080/mcp
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "triton": {
      "type": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### GitHub Copilot

Add to your repository's Copilot MCP configuration (Settings > Copilot > Cloud agent):

```json
{
  "mcpServers": {
    "triton": {
      "type": "http",
      "url": "http://localhost:8080/mcp",
      "tools": ["*"]
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "triton": {
      "type": "http",
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

## Docker Deployment

### Build and Run

```bash
docker compose up -d
```

This builds the image, indexes the docs, and starts the server on port 8080.

### Re-index Docs

```bash
docker compose run --rm triton-mcp triton-index
```

### Custom Port

```bash
TRITON_MCP_PORT=9090 docker compose up -d
```

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
pytest tests/ -v

# Run only direct function tests (no server needed)
pytest tests/ -v -k "TestDirectFunctionCalls"

# Run full integration tests (starts MCP server)
pytest tests/ -v -k "TestMCP"
```

## Architecture

```
triton-mcp/
├── src/triton_mcp/
│   ├── config.py          # Constants, backend info, topics
│   ├── crawler.py          # Async BFS web crawler (httpx + BeautifulSoup)
│   ├── indexer.py           # Document chunking + ChromaDB/SQLite indexing
│   ├── search.py            # Semantic/keyword/hybrid search
│   ├── prompts.py           # Triton system & troubleshooting prompts
│   ├── server.py            # FastMCP server (11 tools, 2 resources, 3 prompts)
│   └── scripts/index_docs.py  # CLI entry point for indexing
├── tests/
│   └── test_integration.py  # Integration tests (AAA pattern)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

- **Crawler**: Async BFS using `httpx`, extracts `<main>` content, sections, and code blocks
- **Indexer**: Splits pages into ~512-token chunks (64-token overlap) with `tiktoken`, embeds via ChromaDB's built-in `all-MiniLM-L6-v2`, and indexes in SQLite FTS5 (Porter stemmer)
- **Search**: Semantic (cosine similarity via ChromaDB), Keyword (BM25 via FTS5), Hybrid (Reciprocal Rank Fusion combining both)
- **Server**: FastMCP with streamable-http transport, lifespan-managed search engine

## Why Triton MCP vs. Web Fetch?

### Pros over raw web fetch

- **Semantic search** — finds relevant info even with different terminology (e.g., "how to batch requests" finds the dynamic batching section)
- **Hybrid search (RRF)** — combines semantic + keyword matching for better recall
- **Pre-indexed** — all ~256 pages chunked and embedded locally, no repeated network calls
- **Structured tools** — `list_backends()`, `get_model_config_template()`, `analyze_config()`, `perf_test_guide()`, etc. generate tailored output instead of reading raw HTML
- **Offline capable** — runs locally after indexing, no internet needed
- **No LLM dependency** — all tools are deterministic, no extra API calls

### Cons vs. raw web fetch

- **Can be stale** — index reflects docs at `triton-index` time; re-run to pick up updates
- **Setup required** — needs `triton-index` and ChromaDB/SQLite dependencies
- **Limited scope** — only covers indexed sources; niche topics may be missing
- **Disk usage** — index at `~/.triton_mcp_index/` takes local storage

### When to use which

| Scenario | Use |
|----------|-----|
| "What backend should I use for X?" | **Triton MCP** (semantic search) |
| "Generate a config.pbtxt for TensorRT LLM" | **Triton MCP** (template generation) |
| "Analyze this config and suggest optimizations" | **Triton MCP** (structured analysis) |
| "Give me perf_analyzer commands for my model" | **Triton MCP** (tailored commands) |
| You know the exact doc URL and need the content | **Web fetch** (always current) |
| Docs have been updated since last index | **Web fetch** (live content) |
| You need something not in the indexed sources | **Web fetch** (unlimited scope) |
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .config import BACKEND_INFO, TOPICS
from .prompts import DEPLOYMENT_GUIDE_TOPICS as DEPLOY_TOPICS
from .prompts import (
    TRITON_OPTIMIZER_PROMPT,
    TRITON_SYSTEM_PROMPT,
    TRITON_TROUBLESHOOTER_PROMPT,
)
from .search import SearchEngine

logger = logging.getLogger(__name__)

_engine: SearchEngine | None = None


def _get_engine() -> SearchEngine:
    global _engine
    if _engine is None or _engine.sqlite_conn is None:
        _engine = SearchEngine()
    return _engine


def _format_results(results: list, max_content_length: int = 3000) -> str:
    if not results:
        return "No results found."
    output: list[str] = []
    for i, r in enumerate(results, 1):
        output.append(f"### Result {i}: {r.page_title}")
        if r.section:
            output.append(f"**Section:** {r.section}")
        output.append(f"**URL:** {r.page_url}")
        output.append(f"**Relevance Score:** {r.score:.3f}")
        content = r.content[:max_content_length]
        if len(r.content) > max_content_length:
            content += "..."
        output.append(content)
        output.append("---")
    return "\n\n".join(output)


@asynccontextmanager
async def app_lifespan(server: FastMCP):
    logger.info("Triton MCP server starting...")
    engine = _get_engine()
    try:
        pages = engine.list_pages()
        logger.info(f"Index ready with {len(pages)} pages")
    except Exception:
        logger.warning(
            "Index not found. Run `triton-index` first to build the docs index."
        )
    yield
    global _engine
    _engine.close()
    _engine = None
    logger.info("Triton MCP server shutting down...")


mcp = FastMCP(
    "Triton Inference Server Docs",
    lifespan=app_lifespan,
    instructions=TRITON_SYSTEM_PROMPT,
)


@mcp.tool()
def search_docs(query: str, mode: str = "hybrid", k: int = 5) -> str:
    """Search NVIDIA Triton Inference Server documentation.

    Args:
        query: Search query (e.g., "dynamic batching configuration", "TensorRT backend setup")
        mode: Search mode - "semantic" (vector similarity), "keyword" (BM25 full-text), or "hybrid" (RRF combination, default)
        k: Number of results to return (default: 5, max: 20)

    Returns:
        Relevant documentation sections with URLs.
    """
    k = min(max(k, 1), 20)
    engine = _get_engine()

    if mode == "semantic":
        results = engine.semantic_search(query, k=k)
    elif mode == "keyword":
        results = engine.keyword_search(query, k=k)
    else:
        results = engine.hybrid_search(query, k=k)

    return _format_results(results)


@mcp.tool()
def get_page(url: str) -> str:
    """Get the full content of a specific Triton documentation page.

    Args:
        url: The URL of the documentation page (from search results)

    Returns:
        Full content of the page, all chunks concatenated.
    """
    engine = _get_engine()
    chunks = engine.get_page(url)
    if not chunks:
        return f"No content found for URL: {url}"
    parts: list[str] = [f"# {chunks[0].page_title}\n"]
    for chunk in chunks:
        parts.append(chunk.content)
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def list_backends() -> str:
    """List all available Triton Inference Server backends with descriptions.

    Returns:
        Table of backends with name, description, and key config fields.
    """
    lines = ["# Triton Inference Server Backends\n"]
    for key, info in BACKEND_INFO.items():
        lines.append(f"## {info['name']} (`{key}`)")
        lines.append(f"{info['description']}\n")
        if info["config_fields"]:
            lines.append(
                "Key config fields: "
                + ", ".join(f"`{f}`" for f in info["config_fields"])
            )
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def get_model_config_template(
    backend: str,
    model_name: str = "my_model",
    max_batch_size: int = 0,
    instance_count: int = 1,
    gpu: bool = True,
) -> str:
    """Generate a config.pbtxt template for a Triton model deployment.

    Args:
        backend: Backend type (tensorrt_llm, vllm, python, pytorch, onnxruntime, tensorrt, fil, dali, custom)
        model_name: Name for the model (default: my_model)
        max_batch_size: Maximum batch size, 0 for dynamic batching with LLMs (default: 0)
        instance_count: Number of model instances per GPU (default: 1)
        gpu: Whether to use GPU (default: true)

    Returns:
        A config.pbtxt template that can be customized for your model.
    """
    backend = backend.lower().strip()
    if backend not in BACKEND_INFO:
        available = ", ".join(f"`{k}`" for k in BACKEND_INFO)
        return f"Unknown backend: {backend}. Available: {available}"

    info = BACKEND_INFO[backend]
    kind = "KIND_GPU" if gpu else "KIND_CPU"
    device = "KIND_GPU" if gpu else "KIND_CPU"

    platform_map = {
        "tensorrt_llm": "tensorrt_llm",
        "vllm": "vllm",
        "python": "python",
        "pytorch": "pytorch_libtorch",
        "onnxruntime": "onnxruntime_onnx",
        "tensorrt": "tensorrt_plan",
        "fil": "fil",
        "dali": "dali",
        "custom": "custom",
    }

    platform = platform_map.get(backend, "custom")

    lines = [
        f'name: "{model_name}"',
        f'platform: "{platform}"',
        f"max_batch_size: {max_batch_size}",
        "",
    ]

    if backend == "tensorrt_llm":
        lines.extend(
            [
                "# TensorRT-LLM specific configuration",
                "# See: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html",
                "",
                "model_parameters: {",
                '  key: "tensor_parallel_size"',
                '  value: { string_value: "1" }',
                "}",
                "model_parameters: {",
                '  key: "max_batch_size"',
                f'  value: {{ string_value: "{max_batch_size}" }}',
                "}",
            ]
        )
    elif backend == "vllm":
        lines.extend(
            [
                "# vLLM backend configuration",
                "# See: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html",
                "",
                "model_parameters: {",
                '  key: "model"',
                '  value: { string_value: "<HUGGING_FACE_MODEL_ID_OR_PATH>" }',
                "}",
            ]
        )
    elif backend == "python":
        lines.extend(
            [
                "# Python backend configuration",
                "# See: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html",
                "",
                "# Place your model.py in the model directory:",
                "# models/<model_name>/1/model.py",
                "",
                "parameters: {",
                '  key: "EXECUTION_ENV_PATH"',
                '  value: { string_value: "/opt/tritonserver/python_backend_stubs/python_311_stub" }',
                "}",
            ]
        )
    else:
        lines.extend(
            [
                "# Input and output tensors",
                "# Customize these based on your model's actual inputs/outputs:",
                "input [",
                "  {",
                '    name: "input"',
                "    data_type: TYPE_FP32",
                "    dims: [-1]",
                "  }",
                "]",
                "output [",
                "  {",
                '    name: "output"',
                "    data_type: TYPE_FP32",
                "    dims: [-1]",
                "  }",
                "]",
            ]
        )

    lines.extend(
        [
            "",
            "instance_group [",
            "  {",
            f"    count: {instance_count}",
            f"    kind: {device}",
            "  }",
            "]",
        ]
    )

    if max_batch_size > 0:
        lines.extend(
            [
                "",
                "# Dynamic batching configuration",
                "dynamic_batching {",
                "  preferred_batch_size: [4, 8]",
                "  max_queue_delay_microseconds: 100000",
                "}",
            ]
        )

    lines.append("")
    return "\n".join(lines)


@mcp.tool()
def get_deployment_guide(topic: str) -> str:
    """Get Triton deployment guides and best practices.

    Args:
        topic: Deployment topic - "docker", "kubernetes", "model_repository", or "performance"

    Returns:
        Deployment guide content for the specified topic.
    """
    topic = topic.lower().strip()
    if topic in DEPLOY_TOPICS:
        return DEPLOY_TOPICS[topic]
    available = ", ".join(f'"{k}"' for k in DEPLOY_TOPICS)
    return f"Unknown topic: {topic}. Available topics: {available}"


@mcp.tool()
def best_practices(topic: str) -> str:
    """Get Triton best practices for a specific topic.

    Args:
        topic: Best practices topic - "architecture", "backends", "model_config", "deployment", "performance", "client", "llm", "protocol"

    Returns:
        Best practices content for the specified topic. Use this to guide proper Triton application development.
    """
    engine = _get_engine()
    topic = topic.lower().strip()

    if topic not in TOPICS:
        available = ", ".join(f'"{k}"' for k in TOPICS)
        return f"Unknown topic: {topic}. Available topics: {available}"

    search_query = f"Triton {topic} best practices {TOPICS[topic]}"
    results = engine.hybrid_search(search_query, k=8)
    if not results:
        return f"No best practices found for topic: {topic}"

    output = [f"# Triton Best Practices: {topic.title()}\n"]
    seen_urls: set[str] = set()
    for r in results:
        if r.page_url not in seen_urls:
            seen_urls.add(r.page_url)
            output.append(f"- [{r.page_title}]({r.page_url})")
    output.append("")
    output.append(_format_results(results, max_content_length=2000))
    return "\n\n".join(output)


@mcp.tool()
def list_doc_pages() -> str:
    """List all indexed Triton documentation pages.

    Returns:
        A list of all page titles and URLs in the index.
    """
    engine = _get_engine()
    pages = engine.list_pages()
    if not pages:
        return "No pages indexed. Run `triton-index` to build the index first."
    lines = [f"# Indexed Triton Documentation ({len(pages)} pages)\n"]
    for p in pages:
        lines.append(f"- [{p['title']}]({p['url']})")
    return "\n".join(lines)


@mcp.tool()
def analyze_config(config_pbtxt: str) -> str:
    """Analyze a Triton model config.pbtxt and suggest optimizations.

    Parses the config, identifies the backend, checks for common issues,
    and searches indexed documentation for backend-specific best practices.

    Args:
        config_pbtxt: The config.pbtxt content to analyze

    Returns:
        Analysis of the config with optimization suggestions.
    """
    import re as _re

    issues: list[str] = []
    suggestions: list[str] = []
    info: dict[str, str] = {}

    name_m = _re.search(r'name:\s*"([^"]+)"', config_pbtxt)
    platform_m = _re.search(r'platform:\s*"([^"]+)"', config_pbtxt)
    max_batch_m = _re.search(r"max_batch_size:\s*(\d+)", config_pbtxt)
    batch_m = _re.search(r"max_batch_size:\s*(\d+)", config_pbtxt)
    has_dynamic_batching = "dynamic_batching" in config_pbtxt
    has_instance_group = "instance_group" in config_pbtxt
    has_response_cache = "response_cache" in config_pbtxt

    if name_m:
        info["name"] = name_m.group(1)
    if platform_m:
        info["platform"] = platform_m.group(1)
    if max_batch_m:
        batch_size = int(max_batch_m.group(1))
        info["max_batch_size"] = str(batch_size)

    backend_hint = ""
    if platform_m:
        plat = platform_m.group(1).lower()
        backend_map = {
            "tensorrt_plan": "tensorrt",
            "tensorrt_llm": "tensorrt_llm",
            "pytorch_libtorch": "pytorch",
            "onnxruntime_onnx": "onnxruntime",
            "python": "python",
            "vllm": "vllm",
            "fil": "fil",
            "dali": "dali",
        }
        for k, v in backend_map.items():
            if k in plat:
                backend_hint = v
                info["backend"] = v
                break

    if "model_parameters" in config_pbtxt:
        if "tensor_parallel" in config_pbtxt:
            backend_hint = backend_hint or "tensorrt_llm"
            info["backend"] = backend_hint

    if not name_m:
        issues.append("Missing 'name' field — every model config must have a name")

    if not platform_m and "model_parameters" not in config_pbtxt:
        issues.append(
            "Missing 'platform' field — specify the backend type (e.g., onnxruntime_onnx, pytorch_libtorch)"
        )

    max_batch_val = int(max_batch_m.group(1)) if max_batch_m else None
    if max_batch_val is not None:
        if max_batch_val == 0 and backend_hint not in ("tensorrt_llm", "vllm"):
            if not has_dynamic_batching:
                suggestions.append(
                    "max_batch_size=0 with dynamic_batching is recommended for throughput — consider adding a dynamic_batching block"
                )
        elif max_batch_val > 0:
            if not has_dynamic_batching:
                suggestions.append(
                    f"max_batch_size={max_batch_val} but no dynamic_batching configured — adding dynamic_batching can improve throughput by grouping requests"
                )
            if max_batch_val > 64 and backend_hint in ("pytorch", "python"):
                suggestions.append(
                    f"max_batch_size={max_batch_val} is very high for {backend_hint} — consider reducing to 8-32 and using multiple instances"
                )

    if not has_instance_group:
        suggestions.append(
            "No instance_group configured — add instance_group with count and kind (KIND_GPU/KIND_CPU) for explicit GPU/CPU placement"
        )
    else:
        if "KIND_GPU" not in config_pbtxt and "KIND_CPU" not in config_pbtxt:
            suggestions.append(
                "instance_group present but kind not specified — add kind: KIND_GPU or KIND_CPU"
            )

    if has_dynamic_batching:
        if "preferred_batch_size" not in config_pbtxt:
            suggestions.append(
                "dynamic_batching present but preferred_batch_size not set — adding preferred_batch_size helps optimize batching"
            )
        if "max_queue_delay_microseconds" not in config_pbtxt:
            suggestions.append(
                "Consider setting max_queue_delay_microseconds in dynamic_batching to control the tradeoff between throughput and latency"
            )

    if not has_response_cache and backend_hint not in ("tensorrt_llm", "vllm"):
        suggestions.append(
            "Consider enabling response_cache for models with repeated inputs to reduce inference latency"
        )

    output = ["# Config Analysis\n"]
    if info:
        output.append("## Model Info")
        for k, v in info.items():
            output.append(f"- **{k}**: {v}")
        output.append("")

    if issues:
        output.append("## Issues")
        for issue in issues:
            output.append(f"- {issue}")
        output.append("")

    if suggestions:
        output.append("## Optimization Suggestions")
        for s in suggestions:
            output.append(f"- {s}")
        output.append("")

    engine = _get_engine()
    if backend_hint:
        search_query = f"Triton {backend_hint} backend configuration best practices"
        results = engine.hybrid_search(search_query, k=3)
        if results:
            output.append("## Relevant Documentation")
            for r in results[:3]:
                output.append(f"- [{r.page_title}]({r.page_url})")

    return "\n".join(output)


@mcp.tool()
def python_client_help(task: str) -> str:
    """Get help with the Triton Python client library (tritonclient).

    Searches indexed client source code and examples for relevant patterns.

    Args:
        task: What you need help with (e.g., "inference request", "shared memory", "async client", "model metadata")

    Returns:
        Python client code examples and API documentation from the indexed sources.
    """
    engine = _get_engine()

    queries = [
        f"tritonclient {task}",
        f"python client {task} example",
    ]

    all_results: list = []
    seen_ids: set[str] = set()
    for q in queries:
        results = engine.hybrid_search(q, k=5)
        for r in results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                all_results.append(r)

    output = [f"# Python Client Help: {task}\n"]

    gh_results = [r for r in all_results if "github.com" in r.page_url]
    doc_results = [r for r in all_results if "github.com" not in r.page_url]

    if gh_results:
        output.append("## Client Source & Examples")
        for r in gh_results[:5]:
            output.append(f"### {r.page_title}")
            output.append(f"**Source:** {r.page_url}")
            content_preview = r.content[:2000]
            output.append(content_preview)
            output.append("---")

    if doc_results:
        output.append("\n## Documentation")
        for r in doc_results[:3]:
            output.append(f"- [{r.page_title}]({r.page_url})")

    if not all_results:
        output.append(
            "No results found. Try a different query, e.g., 'inference', 'shared memory', 'async'."
        )

    return "\n\n".join(output)


@mcp.tool()
def perf_test_guide(
    model_name: str, backend: str = "", protocol: str = "grpc", concurrency: int = 1
) -> str:
    """Generate a performance testing guide using perf_analyzer and model_analyzer.

    Args:
        model_name: Name of the model to benchmark
        backend: Backend type (tensorrt, pytorch, onnxruntime, python, tensorrt_llm, vllm, etc.)
        protocol: Inference protocol — "grpc" or "http" (default: grpc)
        concurrency: Number of concurrent connections (default: 1)

    Returns:
        Step-by-step perf_analyzer and model_analyzer commands with explanations.
    """
    engine = _get_engine()

    output = [f"# Performance Testing Guide for `{model_name}`\n"]

    if backend:
        output.append(f"**Backend:** {backend}")
    output.append(f"**Protocol:** {protocol}")
    output.append(f"**Concurrency:** {concurrency}\n")

    output.append("## 1. Quick Benchmark with perf_analyzer\n")
    output.append("```bash")
    if backend in ("tensorrt_llm", "vllm"):
        output.append("# For LLM models, use genai-perf instead")
        output.append(f"genai-perf -m {model_name} --backend triton \\")
        output.append("  --endpoint-type stream \\")
        output.append(f"  --concurrency {concurrency}")
    else:
        output.append(f"perf_analyzer -m {model_name} \\")
        output.append(f"  -p {protocol} \\")
        output.append(f"  -i {concurrency}")
        if backend == "python":
            output.append("  --shape INPUT:1")
    output.append("```\n")

    output.append("## 2. Throughput & Latency Sweep\n")
    output.append("```bash")
    output.append(f"perf_analyzer -m {model_name} -p {protocol} \\")
    output.append("  --concurrency-range 1:8:2 \\")
    output.append("  --measurement-interval 3000 \\")
    output.append("  --max-threads 4")
    output.append("```\n")

    output.append("## 3. Model Analyzer Profiling\n")
    output.append("```bash")
    output.append("model-analyzer profile \\")
    output.append("  --model-repository /models \\")
    output.append(f"  --profile-models {model_name} \\")
    output.append("  --concurrency 1,2,4,8 \\")
    output.append("  --batch-sizes 1,8,16,32")
    output.append("```\n")

    output.append("## 4. Model Analyzer Analysis\n")
    output.append("```bash")
    output.append("model-analyzer analyze \\")
    output.append(f"  --analysis-models {model_name} \\")
    output.append("  --analyzer-model-config-search \\")
    output.append("  --output /tmp/model_analyzer_results")
    output.append("```\n")

    output.append("## Key Metrics to Watch\n")
    output.append("- **Throughput** (infer/sec): Higher is better")
    output.append("- **Latency p99**: Critical for real-time serving")
    output.append("- **GPU Utilization**: Should be >80% for optimal throughput")
    output.append("- **GPU Memory**: Monitor to avoid OOM\n")

    if backend in ("tensorrt_llm", "vllm"):
        output.append("## LLM-Specific Metrics\n")
        output.append(
            "- **Time to First Token (TTFT)**: End-to-end latency for first token"
        )
        output.append("- **Inter-Token Latency**: Time between generated tokens")
        output.append("- **Output Throughput**: Tokens per second\n")

    tip_query = f"perf_analyzer model_analyzer {backend} performance benchmark"
    results = engine.hybrid_search(tip_query, k=5)
    if results:
        output.append("## Relevant Documentation\n")
        for r in results[:5]:
            output.append(f"- [{r.page_title}]({r.page_url})")

    return "\n".join(output)


@mcp.tool()
def model_optimization_guide(backend: str, model_name: str = "my_model") -> str:
    """Get optimization recommendations for a Triton model deployment.

    Searches indexed documentation for backend-specific optimization tips
    and generates a comprehensive optimization guide.

    Args:
        backend: Backend type (tensorrt, tensorrt_llm, vllm, python, pytorch, onnxruntime, fil, dali)
        model_name: Name of the model (default: my_model)

    Returns:
        Optimization guide with config suggestions and performance tips.
    """
    engine = _get_engine()
    backend = backend.lower().strip()

    if backend not in BACKEND_INFO:
        available = ", ".join(f"`{k}`" for k in BACKEND_INFO)
        return f"Unknown backend: {backend}. Available: {available}"

    info = BACKEND_INFO[backend]

    output = [f"# Optimization Guide: {info['name']} — `{model_name}`\n"]
    output.append(f"{info['description']}\n")

    output.append("## Configuration Optimization\n")

    if backend == "tensorrt":
        output.extend(
            [
                "### TensorRT Optimization",
                "1. **Use INT8/FP16 precision** when accuracy allows — can 2-4x throughput",
                "2. **Set `acceleration: true`** in config.pbtxt to enable TensorRT acceleration",
                "3. **Use dynamic batching** with `preferred_batch_size` matching your throughput needs",
                "4. **Multiple model instances** via `instance_group` for GPU parallelism",
                "5. **Pre-plan the engine** — first inference is slower; warm up with a dummy request",
                "```",
                f'name: "{model_name}"',
                'platform: "tensorrt_plan"',
                "max_batch_size: 32",
                "",
                "dynamic_batching {",
                "  preferred_batch_size: [8, 16, 32]",
                "  max_queue_delay_microseconds: 50000",
                "}",
                "",
                "instance_group [",
                "  { count: 2 kind: KIND_GPU }",
                "]",
                "```",
            ]
        )
    elif backend == "tensorrt_llm":
        output.extend(
            [
                "### TensorRT-LLM Optimization",
                "1. **Enable in-flight batching** — critical for LLM throughput",
                "2. **Set `max_batch_size: 0`** to use unlimited batching with KV cache",
                "3. **Use tensor parallelism** for large models on multiple GPUs",
                "4. **Tune `max_queue_delay_microseconds`** for latency/throughput tradeoff",
                "5. **Enable KV cache reuse** with decoupled transaction policy",
            ]
        )
    elif backend == "vllm":
        output.extend(
            [
                "### vLLM Backend Optimization",
                "1. **PagedAttention** is enabled by default — most efficient KV cache management",
                "2. **Continuous batching** handles variable-length sequences efficiently",
                "3. **Multi-LoRA** support for serving multiple adapters from one model",
                "4. **Set appropriate `gpu_memory_utilization`** (default 0.9) to avoid OOM",
            ]
        )
    elif backend == "pytorch":
        output.extend(
            [
                "### PyTorch Backend Optimization",
                "1. **Use TorchScript** for faster inference vs eager mode",
                "2. **Enable dynamic batching** for throughput gains",
                "3. **Use `KIND_GPU` instance group** for GPU acceleration",
                "4. **Consider model warmup** — first inference may be slower due to JIT compilation",
            ]
        )
    elif backend == "python":
        output.extend(
            [
                "### Python Backend Optimization",
                "1. **Use async `execute`** method with `asyncio` for I/O-bound models",
                "2. **BLS (Business Logic Scripting)** for multi-model pipelines",
                "3. **Avoid blocking calls** in `execute()` — use `await` for async operations",
                "4. **Use shared memory** for large tensor transfers between models",
            ]
        )
    elif backend == "onnxruntime":
        output.extend(
            [
                "### ONNX Runtime Optimization",
                "1. **Use GPU execution provider** (`KIND_GPU`) for CUDA acceleration",
                "2. **Enable dynamic batching** for throughput",
                "3. **Convert models to ONNX from PyTorch/TensorFlow** for best compatibility",
                "4. **Set `acceleration: true`** to enable ORT acceleration",
            ]
        )
    else:
        output.append(f"See the {info['name']} documentation for optimization tips.")

    # Search for specific optimization docs
    search_query = f"{info['name']} optimization performance tuning {model_name}"
    results = engine.hybrid_search(search_query, k=5)
    if results:
        output.append("\n## Relevant Documentation\n")
        for r in results[:5]:
            output.append(f"- [{r.page_title}]({r.page_url})")

    return "\n".join(output)


@mcp.resource("triton://docs/index")
def get_docs_index() -> str:
    """Index of all Triton documentation sections."""
    engine = _get_engine()
    pages = engine.list_pages()
    if not pages:
        return "No pages indexed."
    return json.dumps(pages, indent=2)


@mcp.resource("triton://backends")
def get_backends_resource() -> str:
    """List of available Triton backends."""
    return json.dumps(
        {
            k: {"name": v["name"], "description": v["description"]}
            for k, v in BACKEND_INFO.items()
        },
        indent=2,
    )


@mcp.prompt(title="Triton Developer")
def developer(task: str) -> str:
    """Prompt for building Triton applications. Provides system context and asks for help with a specific task."""
    return f"""{TRITON_SYSTEM_PROMPT}

I need help with the following Triton task: {task}

Please search the documentation first with search_docs, then provide a detailed answer with code examples and config.pbtxt where appropriate."""


@mcp.prompt(title="Triton Troubleshooter")
def troubleshooter(error_message: str, model_name: str = "", backend: str = "") -> str:
    """Prompt for debugging Triton deployment issues. Provides troubleshooting context and asks for help with a specific error."""
    context = f"""{TRITON_TROUBLESHOOTER_PROMPT}

I'm encountering the following error with my Triton deployment:

Error: {error_message}"""

    if model_name:
        context += f"\nModel: {model_name}"
    if backend:
        context += f"\nBackend: {backend}"

    context += "\n\nPlease search the documentation for troubleshooting guidance and suggest specific fixes."

    return context


@mcp.prompt(title="Triton Optimizer")
def optimizer(model_name: str, backend: str, current_config: str = "") -> str:
    """Prompt for optimizing Triton model performance. Provides optimization context and asks for help with a specific model."""
    context = f"""{TRITON_OPTIMIZER_PROMPT}

I need to optimize the performance of my Triton model:
- **Model:** {model_name}
- **Backend:** {backend}"""

    if current_config:
        context += f"\n- **Current config.pbtxt:**\n```\n{current_config}\n```"

    context += "\n\nPlease use model_optimization_guide and analyze_config to help optimize this deployment."

    return context


def main():
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("TRITON_MCP_PORT", "8080"))
    host = os.environ.get("TRITON_MCP_HOST", "0.0.0.0")
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

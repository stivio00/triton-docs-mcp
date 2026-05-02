from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import pytest
import requests

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_HOST = "127.0.0.1"
MCP_PORT = 9876
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

EXPECTED_TOOLS = [
    "search_docs",
    "get_page",
    "list_backends",
    "get_model_config_template",
    "get_deployment_guide",
    "best_practices",
    "list_doc_pages",
    "analyze_config",
    "python_client_help",
    "perf_test_guide",
    "model_optimization_guide",
]

EXPECTED_BACKENDS = [
    "tensorrt_llm",
    "vllm",
    "python",
    "pytorch",
    "onnxruntime",
    "tensorrt",
    "fil",
    "dali",
    "custom",
]

DEPLOY_TOPICS = ["docker", "kubernetes", "model_repository", "performance"]

BEST_PRACTICE_TOPICS = [
    "architecture",
    "backends",
    "model_config",
    "deployment",
    "performance",
    "client",
    "llm",
    "protocol",
]


@pytest.fixture(scope="module")
def mcp_server():
    env = os.environ.copy()
    env["TRITON_DOCS_MCP_PORT"] = str(MCP_PORT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "triton_docs_mcp.server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)
    for _ in range(30):
        try:
            resp = requests.get(f"http://{MCP_HOST}:{MCP_PORT}/", timeout=1)
            if resp.status_code in (200, 404, 405):
                break
        except requests.ConnectionError:
            time.sleep(0.5)
    yield proc
    proc.terminate()
    proc.wait(timeout=10)


class TestMCPToolDiscovery:
    """Verify that the MCP server exposes the expected set of tools."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_expected_tools(self, mcp_server):
        """Arrange: connect to the MCP server via streamable HTTP"""
        # Act: list available tools
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

        # Assert: all expected tools are present
        tool_names = [t.name for t in result.tools]
        for expected in EXPECTED_TOOLS:
            assert expected in tool_names, (
                f"Expected tool '{expected}' not found in {tool_names}"
            )

    @pytest.mark.asyncio
    async def test_tool_count_matches(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: list tools
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()

        # Assert: exactly 7 tools registered
        assert len(result.tools) == len(EXPECTED_TOOLS)


class TestMCPToolExecution:
    """Verify that individual MCP tools return correct results."""

    @pytest.mark.asyncio
    async def test_list_backends_returns_all_backends(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: call list_backends tool
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("list_backends", {})

        # Assert: all backends are listed
        text = result.content[0].text
        for backend in EXPECTED_BACKENDS:
            assert backend in text, (
                f"Backend '{backend}' not found in list_backends output"
            )

    @pytest.mark.asyncio
    async def test_search_docs_hybrid_returns_results(self, mcp_server):
        """Arrange: connect to MCP server with a known query"""
        # Act: search for dynamic batching
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_docs",
                    {
                        "query": "dynamic batching",
                        "mode": "hybrid",
                        "k": 3,
                    },
                )

        # Assert: results contain relevant content
        text = result.content[0].text
        assert "Result" in text or "batch" in text.lower(), (
            f"Search results don't contain expected content: {text[:200]}"
        )

    @pytest.mark.asyncio
    async def test_search_docs_semantic_returns_results(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: search using semantic mode
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_docs",
                    {
                        "query": "TensorRT model deployment",
                        "mode": "semantic",
                        "k": 2,
                    },
                )

        # Assert: results are returned
        text = result.content[0].text
        assert len(text) > 50, (
            f"Semantic search returned too little content: {text[:100]}"
        )

    @pytest.mark.asyncio
    async def test_search_docs_keyword_returns_results(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: search using keyword mode
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_docs",
                    {
                        "query": "model repository",
                        "mode": "keyword",
                        "k": 2,
                    },
                )

        # Assert: results contain the search terms
        text = result.content[0].text
        assert "model" in text.lower() and "repository" in text.lower(), (
            f"Keyword search results don't contain expected terms: {text[:200]}"
        )

    @pytest.mark.asyncio
    async def test_get_model_config_template_python(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: generate config for python backend
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_model_config_template",
                    {
                        "backend": "python",
                        "model_name": "my_python_model",
                        "max_batch_size": 8,
                    },
                )

        # Assert: config contains required fields
        text = result.content[0].text
        assert 'name: "my_python_model"' in text, "Config should contain model name"
        assert 'platform: "python"' in text, "Config should contain python platform"
        assert "max_batch_size: 8" in text, "Config should contain max_batch_size"
        assert "instance_group" in text, "Config should contain instance_group"
        assert "dynamic_batching" in text, (
            "Config should contain dynamic_batching for batch_size > 0"
        )

    @pytest.mark.asyncio
    async def test_get_model_config_template_tensorrt_llm(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: generate config for tensorrt_llm backend with LLM defaults
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_model_config_template",
                    {
                        "backend": "tensorrt_llm",
                        "model_name": "llama_model",
                        "max_batch_size": 0,
                    },
                )

        # Assert: config has LLM-specific fields and no dynamic batching for batch_size=0
        text = result.content[0].text
        assert 'name: "llama_model"' in text
        assert "tensorrt_llm" in text
        assert "dynamic_batching" not in text, (
            "LLM with batch_size=0 should not have dynamic_batching"
        )

    @pytest.mark.asyncio
    async def test_get_model_config_template_unknown_backend(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: request config for an unknown backend
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_model_config_template",
                    {
                        "backend": "nonexistent_backend",
                    },
                )

        # Assert: returns error about unknown backend
        text = result.content[0].text
        assert "Unknown backend" in text, "Should report unknown backend"
        assert "nonexistent_backend" in text

    @pytest.mark.asyncio
    async def test_get_deployment_guide_docker(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: request docker deployment guide
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_deployment_guide", {"topic": "docker"}
                )

        # Assert: docker guide contains key information
        text = result.content[0].text
        assert "docker" in text.lower()
        assert "nvcr.io" in text or "tritonserver" in text, (
            "Docker guide should mention NGC container"
        )

    @pytest.mark.asyncio
    async def test_get_deployment_guide_unknown_topic(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: request unknown deployment topic
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "get_deployment_guide", {"topic": "nonexistent"}
                )

        # Assert: returns available topics
        text = result.content[0].text
        assert "Unknown topic" in text
        for topic in DEPLOY_TOPICS:
            assert topic in text, f"Available topic '{topic}' should be listed"

    @pytest.mark.asyncio
    async def test_best_practices_valid_topic(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: request best practices for architecture
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "best_practices", {"topic": "architecture"}
                )

        # Assert: returns best practices with links
        text = result.content[0].text
        assert "Architecture" in text or "architecture" in text.lower()
        assert "docs.nvidia.com" in text or "Result" in text, (
            "Should reference docs or show search results"
        )

    @pytest.mark.asyncio
    async def test_best_practices_invalid_topic(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: request best practices for invalid topic
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "best_practices", {"topic": "invalid_topic"}
                )

        # Assert: lists available topics
        text = result.content[0].text
        assert "Unknown topic" in text
        for topic in BEST_PRACTICE_TOPICS:
            assert topic in text, f"Topic '{topic}' should be listed as available"

    @pytest.mark.asyncio
    async def test_list_doc_pages_returns_pages(self, mcp_server):
        """Arrange: connect to MCP server (index already built)"""
        # Act: list all indexed pages
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("list_doc_pages", {})

        # Assert: returns non-empty page list
        text = result.content[0].text
        assert "Indexed Triton Documentation" in text
        assert "docs.nvidia.com" in text, "Pages should reference the NVIDIA docs URL"

    @pytest.mark.asyncio
    async def test_get_page_with_valid_url(self, mcp_server):
        """Arrange: first get a URL from list_doc_pages"""
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                list_result = await session.call_tool("list_doc_pages", {})
                pages_text = list_result.content[0].text

                # Extract first URL from the page list
                urls = re.findall(r"\((https://docs\.nvidia\.com[^)]+)\)", pages_text)
                assert len(urls) > 0, "Should have at least one page URL"

                # Act: get page content
                page_result = await session.call_tool("get_page", {"url": urls[0]})

        # Assert: page content is non-empty and substantial
        text = page_result.content[0].text
        assert len(text) > 100, (
            f"Page content should be substantial, got {len(text)} chars"
        )


class TestMCPResources:
    """Verify MCP resources return correct data."""

    @pytest.mark.asyncio
    async def test_list_resources(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: list registered resources
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                resources = await session.list_resources()

        # Assert: triton://docs/index and triton://backends resources are registered
        uris = [str(r.uri) for r in resources.resources]
        assert "triton://docs/index" in uris, (
            f"Expected triton://docs/index in resources: {uris}"
        )
        assert "triton://backends" in uris, (
            f"Expected triton://backends in resources: {uris}"
        )

    @pytest.mark.asyncio
    async def test_backends_resource_returns_json(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: read the backends resource
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.read_resource("triton://backends")

        # Assert: returns valid JSON with backend info
        text = result.contents[0].text
        data = json.loads(text)
        assert "tensorrt_llm" in data, "Expected tensorrt_llm in backends resource"
        assert "python" in data, "Expected python in backends resource"
        assert data["tensorrt_llm"]["name"] == "TensorRT-LLM"


class TestMCPPrompts:
    """Verify MCP prompts are registered and functional."""

    @pytest.mark.asyncio
    async def test_list_prompts(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: list available prompts
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_prompts()

        # Assert: expected prompts are registered (MCP uses function names, not titles)
        prompt_names = [p.name for p in result.prompts]
        assert "developer" in prompt_names, (
            f"Expected 'developer' prompt, got: {prompt_names}"
        )
        assert "troubleshooter" in prompt_names, (
            f"Expected 'troubleshooter' prompt, got: {prompt_names}"
        )

    @pytest.mark.asyncio
    async def test_developer_prompt(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: get the developer prompt with a task
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.get_prompt(
                    "developer", {"task": "deploy a PyTorch model"}
                )

        # Assert: prompt contains system instructions and the task
        text = "\n".join(
            [m.content.text for m in result.messages if hasattr(m.content, "text")]
        )
        assert "Triton" in text
        assert "deploy a PyTorch model" in text

    @pytest.mark.asyncio
    async def test_troubleshooter_prompt(self, mcp_server):
        """Arrange: connect to MCP server"""
        # Act: get the troubleshooter prompt with an error
        async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.get_prompt(
                    "troubleshooter",
                    {
                        "error_message": "model load failed",
                        "model_name": "my_model",
                        "backend": "pytorch",
                    },
                )

        # Assert: prompt contains the error and model info
        text = "\n".join(
            [m.content.text for m in result.messages if hasattr(m.content, "text")]
        )
        assert "model load failed" in text
        assert "my_model" in text


class TestDirectFunctionCalls:
    """Test tool functions directly without MCP transport for faster feedback."""

    def test_search_docs_direct(self):
        """Arrange: get the search engine directly"""
        from triton_docs_mcp.search import SearchEngine

        engine = SearchEngine()

        # Act: search for model configuration
        results = engine.hybrid_search("model configuration config.pbtxt", k=3)

        # Assert: results are returned and contain relevant content
        assert len(results) > 0, "Should return at least one result"
        assert any(
            "model" in r.page_title.lower() or "config" in r.content.lower()
            for r in results
        ), "At least one result should mention model configuration"

        engine.close()

    def test_list_backends_direct(self):
        """Arrange: import backends directly from config"""
        from triton_docs_mcp.config import BACKEND_INFO

        # Act: get the backend keys
        result_backends = list(BACKEND_INFO.keys())

        # Assert: all expected backends are present
        for backend in EXPECTED_BACKENDS:
            assert backend in result_backends, f"Backend '{backend}' missing"

    def test_get_model_config_template_direct(self):
        """Arrange: get the config template function from server module"""
        from triton_docs_mcp.server import get_model_config_template

        # Act: generate config for vLLM backend
        result = get_model_config_template("vllm", model_name="llama_test")

        # Assert: config contains vLLM-specific fields
        assert 'name: "llama_test"' in result
        assert 'platform: "vllm"' in result
        assert "HUGGING_FACE_MODEL_ID_OR_PATH" in result

    def test_search_modes_return_different_results(self):
        """Arrange: create search engine with same query"""
        from triton_docs_mcp.search import SearchEngine

        engine = SearchEngine()
        query = "batching configuration"

        # Act: search with different modes
        semantic = engine.semantic_search(query, k=3)
        keyword = engine.keyword_search(query, k=3)
        hybrid = engine.hybrid_search(query, k=3)

        # Assert: all modes return results
        assert len(semantic) > 0, "Semantic search should return results"
        assert len(keyword) > 0, "Keyword search should return results"
        assert len(hybrid) > 0, "Hybrid search should return results"

        engine.close()

    def test_analyze_config_with_valid_config(self):
        """Arrange: create a typical config.pbtxt"""
        from triton_docs_mcp.server import analyze_config

        config = """name: "my_model"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  { name: "input" data_type: TYPE_FP32 dims: [1, 3, 224, 224] }
]
output [
  { name: "output" data_type: TYPE_FP32 dims: [1000] }
]
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 50000
}
instance_group [
  { count: 1 kind: KIND_GPU }
]"""

        # Act: analyze the config
        result = analyze_config(config)

        # Assert: analysis contains relevant info
        assert "my_model" in result, "Should show model name"
        assert "onnxruntime" in result.lower() or "Optimization" in result, (
            "Should identify backend"
        )

    def test_analyze_config_with_issues(self):
        """Arrange: create an incomplete config.pbtxt"""
        from triton_docs_mcp.server import analyze_config

        config = 'name: "broken_model"\nmax_batch_size: 32'

        # Act: analyze the config
        result = analyze_config(config)

        # Assert: analysis identifies issues
        assert "Missing" in result or "Issue" in result, "Should flag missing fields"

    def test_model_optimization_guide_tensorrt(self):
        """Arrange: request optimization guide for tensorrt backend"""
        from triton_docs_mcp.server import model_optimization_guide

        # Act: get optimization guide
        result = model_optimization_guide("tensorrt", model_name="resnet50")

        # Assert: guide contains TensorRT-specific content
        assert "TensorRT" in result
        assert "resnet50" in result
        assert (
            "dynamic_batching" in result.lower() or "instance_group" in result.lower()
        )

    def test_model_optimization_guide_unknown_backend(self):
        """Arrange: request optimization guide for unknown backend"""
        from triton_docs_mcp.server import model_optimization_guide

        # Act: get optimization guide for unknown backend
        result = model_optimization_guide("nonexistent")

        # Assert: returns error with available backends
        assert "Unknown backend" in result

    def test_perf_test_guide(self):
        """Arrange: request perf test guide"""
        from triton_docs_mcp.server import perf_test_guide

        # Act: get perf test guide
        result = perf_test_guide("my_model", backend="pytorch", protocol="grpc")

        # Assert: guide contains expected sections
        assert "my_model" in result
        assert "perf_analyzer" in result
        assert "model-analyzer" in result or "model_analyzer" in result

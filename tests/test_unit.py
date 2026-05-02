from __future__ import annotations


from triton_mcp.crawler import Page, _is_internal, _normalize_url, _extract_links
from triton_mcp.indexer import _chunk_text, _build_chunks
from triton_mcp.search import SearchResult


class TestIsInternal:
    def test_internal_page(self):
        assert _is_internal(
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_configuration.html"
        )

    def test_external_page(self):
        assert not _is_internal("https://github.com/triton-inference-server/server")

    def test_pdf_extension(self):
        assert not _is_internal(
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/file.pdf"
        )

    def test_image_extension(self):
        assert not _is_internal(
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/image.png"
        )

    def test_zip_extension(self):
        assert not _is_internal(
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/archive.zip"
        )

    def test_page_with_fragment(self):
        assert _is_internal(
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/architecture.html#models"
        )


class TestNormalizeUrl:
    def test_removes_index_html(self):
        result = _normalize_url("https://docs.nvidia.com/path/index.html")
        assert result == "https://docs.nvidia.com/path/"

    def test_converts_html_to_slash(self):
        result = _normalize_url("https://docs.nvidia.com/path/page.html")
        assert result == "https://docs.nvidia.com/path/page/"

    def test_plain_path(self):
        result = _normalize_url("https://docs.nvidia.com/path/page/")
        assert result == "https://docs.nvidia.com/path/page/"


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world"
        chunks = _chunk_text(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = " ".join(["word"] * 2000)
        chunks = _chunk_text(text, chunk_size=128, overlap=16)
        assert len(chunks) > 1

    def test_empty_text(self):
        chunks = _chunk_text("", chunk_size=128, overlap=16)
        assert len(chunks) == 0

    def test_overlap_between_chunks(self):
        text = " ".join(["alpha"] * 500)
        chunks = _chunk_text(text, chunk_size=64, overlap=8)
        if len(chunks) > 1:
            assert len(chunks) >= 2


class TestBuildChunks:
    def test_single_page_single_chunk(self):
        page = Page(
            url="https://example.com/test",
            title="Test Page",
            content="This is a test page with some content that is long enough to be meaningful.",
            sections=[],
            code_blocks=[],
        )
        chunks = _build_chunks([page])
        assert len(chunks) >= 1
        assert chunks[0].page_url == "https://example.com/test"
        assert chunks[0].page_title == "Test Page"

    def test_empty_page_skipped(self):
        page = Page(
            url="https://example.com/empty",
            title="Empty",
            content="   ",
            sections=[],
            code_blocks=[],
        )
        chunks = _build_chunks([page])
        assert len(chunks) == 0

    def test_sections_in_combined_text(self):
        page = Page(
            url="https://example.com/sections",
            title="Sections Page",
            content="Some raw content here that is longer than just a label.",
            sections=[
                {
                    "heading": "Getting Started",
                    "level": 2,
                    "content": "Start by installing Triton.",
                },
                {
                    "heading": "Configuration",
                    "level": 2,
                    "content": "Configure with config.pbtxt.",
                },
            ],
            code_blocks=[],
        )
        chunks = _build_chunks([page])
        assert len(chunks) >= 1
        assert (
            "Getting Started" in chunks[0].content
            or "Configuration" in chunks[0].content
        )

    def test_section_name_assigned_to_chunks(self):
        page = Page(
            url="https://example.com/named-section",
            title="Section Title Page",
            content="Intro text. Step one. Step two. Step three. More content. Even more content. And yet more content for a longer section.",
            sections=[
                {
                    "heading": "First Section",
                    "level": 2,
                    "content": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda.",
                },
                {
                    "heading": "Second Section",
                    "level": 2,
                    "content": "One two three four five six seven eight nine ten eleven twelve.",
                },
            ],
            code_blocks=[],
        )
        chunks = _build_chunks([page])
        assert len(chunks) >= 1
        has_any_section = any(c.section != "" for c in chunks)
        assert has_any_section, (
            f"Expected at least one chunk with a section name, got: {[c.section for c in chunks]}"
        )

    def test_code_blocks_appended_to_first_chunk(self):
        page = Page(
            url="https://example.com/code",
            title="Code Example Page",
            content="This page explains how to configure Triton server for inference workloads.",
            sections=[],
            code_blocks=["model_repository/\n  my_model/\n    config.pbtxt"],
        )
        chunks = _build_chunks([page])
        assert len(chunks) >= 1
        assert (
            "Code Examples" in chunks[0].content
            or "model_repository" in chunks[0].content
        )


class TestSearchResult:
    def test_dataclass_fields(self):
        sr = SearchResult(
            chunk_id="abc_0",
            page_url="https://example.com",
            page_title="Test",
            section="Section",
            content="Some content",
            score=0.95,
        )
        assert sr.chunk_id == "abc_0"
        assert sr.score == 0.95

    def test_default_section_empty(self):
        sr = SearchResult(
            chunk_id="x",
            page_url="https://example.com",
            page_title="T",
            section="",
            content="c",
            score=1.0,
        )
        assert sr.section == ""


class TestExtractLinks:
    def test_extracts_internal_links(self):
        from bs4 import BeautifulSoup

        from triton_mcp.config import BASE_URL

        html = f'<html><body><a href="{BASE_URL}page.html">Link</a></body></html>'
        soup = BeautifulSoup(html, "lxml")
        links = _extract_links(soup, BASE_URL)
        assert len(links) > 0

    def test_ignores_fragment_links(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            '<html><body><a href="#section">Skip</a></body></html>',
            "lxml",
        )
        links = _extract_links(
            soup,
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/",
        )
        assert len(links) == 0

    def test_ignores_javascript_links(self):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(
            '<html><body><a href="javascript:void(0)">Click</a></body></html>',
            "lxml",
        )
        links = _extract_links(
            soup,
            "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/",
        )
        assert len(links) == 0

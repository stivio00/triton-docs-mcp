from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from .config import (
    BASE_URL,
    CRAWL_DELAY_SECONDS,
    CRAWL_MAX_CONCURRENT,
    CRAWL_TIMEOUT_SECONDS,
)
from .config import GITHUB_API_URL, GITHUB_RAW_URL, GITHUB_SOURCES

logger = logging.getLogger(__name__)

_ALLOWED_PREFIX = urlparse(BASE_URL)
_ALLOWED_NETLOC = _ALLOWED_PREFIX.netloc
_ALLOWED_PATH_PREFIX = _ALLOWED_PREFIX.path.rsplit("/", 1)[0]

_EXT_TO_SKIP = {
    ".pdf",
    ".zip",
    ".tar.gz",
    ".gz",
    ".whl",
    ".png",
    ".jpg",
    ".svg",
    ".gif",
    ".mp4",
}


@dataclass
class Page:
    url: str
    title: str
    content: str
    sections: list[dict] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)


def _is_internal(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc and parsed.netloc != _ALLOWED_NETLOC:
        return False
    if parsed.netloc == _ALLOWED_NETLOC and not parsed.path.startswith(
        _ALLOWED_PATH_PREFIX
    ):
        return False
    lower = url.lower()
    for ext in _EXT_TO_SKIP:
        if lower.endswith(ext):
            return False
    if parsed_fragment := parsed.fragment:
        pass
    return True


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path
    if path.endswith("/index.html"):
        path = path[: -len("index.html")]
    elif path.endswith(".html"):
        path = path[:-5] + "/"
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _extract_page(soup: BeautifulSoup, url: str) -> Page | None:
    main = (
        soup.find("main", id="main-content")
        or soup.find("main")
        or soup.find("div", role="main")
    )
    if not main:
        main = (
            soup.find("div", class_=re.compile(r"document|content|body")) or soup.body
        )

    if not main:
        return None

    for tag in main.find_all(
        ["nav", "header", "footer", "aside", "script", "style", "form"]
    ):
        tag.decompose()

    for tag in main.find_all(
        class_=re.compile(r"sidebar|toc|breadcrumb|feedback|related", re.I)
    ):
        tag.decompose()

    title_el = soup.find("h1")
    if title_el:
        title = title_el.get_text(strip=True).rstrip("#").strip()
    else:
        title = (
            urlparse(url)
            .path.split("/")[-1]
            .replace(".html", "")
            .replace("_", " ")
            .title()
        )
        if not title:
            title = urlparse(url).path.split("/")[-2].replace("_", " ").title()

    sections: list[dict] = []
    for heading in main.find_all(re.compile(r"^h[1-6]$")):
        level = int(heading.name[1])
        text = heading.get_text(strip=True)
        if not text:
            continue
        content_parts: list[str] = []
        sibling = heading.find_next_sibling()
        while sibling and sibling.name not in [f"h{i}" for i in range(1, level + 1)]:
            if (
                isinstance(sibling, Tag)
                and sibling.name != "script"
                and sibling.name != "style"
            ):
                content_parts.append(sibling.get_text(separator=" ", strip=True))
            sibling = sibling.find_next_sibling()
        section_text = " ".join(content_parts).strip()
        if section_text:
            sections.append({"heading": text, "level": level, "content": section_text})

    code_blocks: list[str] = []
    for pre in main.find_all("pre"):
        code = pre.find("code")
        text = (code or pre).get_text(strip=True)
        if text and len(text) > 10:
            code_blocks.append(text)

    for tag in main.find_all(["pre", "code"]):
        tag.decompose()

    content = main.get_text(separator="\n", strip=True)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()

    if len(content) < 50:
        return None

    return Page(
        url=url,
        title=title,
        content=content,
        sections=sections,
        code_blocks=code_blocks,
    )


def _extract_links(soup: BeautifulSoup, base_url: str) -> set[str]:
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if (
            href.startswith("#")
            or href.startswith("mailto:")
            or href.startswith("javascript:")
        ):
            continue
        full = urljoin(base_url, href)
        full = full.split("#")[0].split("?")[0]
        if full.startswith(BASE_URL) or full.startswith(BASE_URL.rstrip("/")):
            links.add(full)
    return links


async def crawl(max_pages: int = 0) -> list[Page]:
    visited: set[str] = set()
    queue: asyncio.Queue[str] = asyncio.Queue()
    pages: list[Page] = []
    semaphore = asyncio.Semaphore(CRAWL_MAX_CONCURRENT)
    seen_urls: set[str] = set()

    await queue.put(BASE_URL.rstrip("/") + "/index.html")
    seen_urls.add(_normalize_url(BASE_URL.rstrip("/") + "/index.html"))
    seen_urls.add(_normalize_url(BASE_URL))

    async with httpx.AsyncClient(
        timeout=CRAWL_TIMEOUT_SECONDS, follow_redirects=True
    ) as client:

        async def _fetch(url: str) -> tuple[BeautifulSoup | None, str]:
            async with semaphore:
                await asyncio.sleep(CRAWL_DELAY_SECONDS)
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return BeautifulSoup(resp.text, "lxml"), url
                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    return None, url

        active: set[asyncio.Task] = set()

        while queue.qsize() > 0 or active:
            while not queue.empty() and (max_pages == 0 or len(visited) < max_pages):
                url = await queue.get()
                if url in visited:
                    continue
                visited.add(url)
                task = asyncio.create_task(_fetch(url))
                active.add(task)

            if not active:
                break

            done, active = await asyncio.wait(
                active, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                soup, url = await task
                if soup is None:
                    continue

                page = _extract_page(soup, url)
                if page:
                    pages.append(page)
                    logger.info(f"Indexed: {page.title} ({url})")

                new_links = _extract_links(soup, url)
                for link in new_links:
                    norm = _normalize_url(link)
                    if norm not in seen_urls and _is_internal(link):
                        seen_urls.add(norm)
                        if max_pages == 0 or len(visited) < max_pages:
                            await queue.put(link)

    logger.info(f"Crawl complete: {len(pages)} pages indexed")
    return pages


async def crawl_github_sources() -> list[Page]:
    pages: list[Page] = []
    semaphore = asyncio.Semaphore(3)

    async with httpx.AsyncClient(
        timeout=CRAWL_TIMEOUT_SECONDS, follow_redirects=True
    ) as client:
        for source_name, source in GITHUB_SOURCES.items():
            repo = source["repo"]
            branch = source["branch"]
            all_paths = list(source.get("paths", []))

            for glob_pat in source.get("extra_glob_patterns", []):
                api_url = f"{GITHUB_API_URL}/{repo}/contents/{glob_pat.rsplit('/', 1)[0] if '/' in glob_pat else ''}?ref={branch}"
                try:
                    resp = await client.get(
                        api_url, headers={"Accept": "application/vnd.github.v3+json"}
                    )
                    if resp.status_code == 200:
                        for item in resp.json():
                            if isinstance(item, dict) and item.get("type") == "file":
                                import fnmatch

                                basename = item.get("name", "")
                                if fnmatch.fnmatch(
                                    basename,
                                    glob_pat.rsplit("/", 1)[-1]
                                    if "/" in glob_pat
                                    else glob_pat,
                                ):
                                    all_paths.append(item["path"])
                except Exception as e:
                    logger.warning(f"Failed to list {glob_pat} in {repo}: {e}")

            async def _fetch_github(path: str) -> Page | None:
                url = f"{GITHUB_RAW_URL}/{repo}/{branch}/{path}"
                async with semaphore:
                    await asyncio.sleep(CRAWL_DELAY_SECONDS)
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        text = resp.text
                    except Exception as e:
                        logger.warning(f"Failed to fetch {url}: {e}")
                        return None

                if path.endswith(".md"):
                    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
                    title = (
                        title_match.group(1).strip()
                        if title_match
                        else path.rsplit("/", 1)[-1]
                        .replace(".md", "")
                        .replace("_", " ")
                        .title()
                    )
                    code_blocks: list[str] = []
                    in_code = False
                    code_buf: list[str] = []
                    for line in text.split("\n"):
                        if line.startswith("```"):
                            if in_code and code_buf:
                                code_blocks.append("\n".join(code_buf))
                                code_buf = []
                            in_code = not in_code
                            continue
                        if in_code:
                            code_buf.append(line)
                    sections: list[dict] = []
                    current_heading = ""
                    current_content: list[str] = []
                    for line in text.split("\n"):
                        h_match = re.match(r"^(#{1,6})\s+(.+)$", line)
                        if h_match:
                            if current_heading and current_content:
                                sections.append(
                                    {
                                        "heading": current_heading,
                                        "level": len(h_match.group(1)),
                                        "content": " ".join(current_content),
                                    }
                                )
                            current_heading = h_match.group(2).strip()
                            current_content = []
                        else:
                            current_content.append(line.strip())
                    if current_heading and current_content:
                        sections.append(
                            {
                                "heading": current_heading,
                                "level": 1,
                                "content": " ".join(current_content),
                            }
                        )
                    return Page(
                        url=url,
                        title=title,
                        content=text,
                        sections=sections,
                        code_blocks=code_blocks,
                    )
                elif path.endswith(".py"):
                    title = (
                        path.rsplit("/", 1)[-1]
                        .replace(".py", "")
                        .replace("_", " ")
                        .title()
                    )
                    docstrings: list[str] = []
                    current_doc: list[str] = []
                    in_doc = False
                    for line in text.split("\n"):
                        if '"""' in line or "'''" in line:
                            if in_doc:
                                current_doc.append(line)
                                docstrings.append(
                                    " ".join(current_doc)
                                    .replace('"""', "")
                                    .replace("'''", "")
                                    .strip()
                                )
                                current_doc = []
                                in_doc = False
                            else:
                                in_doc = True
                                current_doc = [line]
                        elif in_doc:
                            current_doc.append(line)
                    classes = re.findall(r"class\s+(\w+)", text)
                    functions = re.findall(r"def\s+(\w+)", text)
                    content_parts = [f"# {title}", f"Source: {repo}/{path}", ""]
                    if classes:
                        content_parts.append(f"Classes: {', '.join(classes)}")
                    if functions:
                        content_parts.append(f"Functions: {', '.join(functions)}")
                    if docstrings:
                        content_parts.append("")
                        content_parts.append("## API Documentation")
                        for doc in docstrings[:20]:
                            content_parts.append(doc)
                    content_parts.append("")
                    content_parts.append("## Source Code")
                    content_parts.append(text)
                    content = "\n".join(content_parts)
                    code_blocks = [text]
                    return Page(
                        url=url,
                        title=f"[{source_name}] {title}",
                        content=content,
                        code_blocks=code_blocks,
                    )
                else:
                    title = (
                        path.rsplit("/", 1)[-1].replace(".", " ").title()
                        if path
                        else source_name
                    )
                    return Page(
                        url=url,
                        title=f"[{source_name}] {title}",
                        content=text,
                        code_blocks=[text] if text else [],
                    )

            for path in all_paths:
                page = await _fetch_github(path)
                if page:
                    pages.append(page)
                    logger.info(f"GitHub [{source_name}]: {page.title}")

    logger.info(f"GitHub crawl complete: {len(pages)} pages")
    return pages

from __future__ import annotations

import asyncio
import logging
import sys

from ..crawler import crawl, crawl_github_sources
from ..indexer import Indexer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    skip_github = "--skip-github" in sys.argv
    skip_docs = "--skip-docs" in sys.argv
    max_pages = 0

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        try:
            max_pages = int(args[0])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [max_pages] [--skip-github] [--skip-docs]")
            print("  max_pages: limit docs pages to crawl (0 = unlimited)")
            print("  --skip-github: skip GitHub source indexing")
            print("  --skip-docs: skip docs site crawling")
            sys.exit(1)

    all_pages = []

    if not skip_docs:
        print("Crawling Triton documentation...")
        docs_pages = asyncio.run(crawl(max_pages=max_pages))
        print(f"Crawled {len(docs_pages)} docs pages")
        all_pages.extend(docs_pages)

    if not skip_github:
        print("Crawling GitHub sources (server, client, perf_analyzer, model_analyzer)...")
        gh_pages = asyncio.run(crawl_github_sources())
        print(f"Crawled {len(gh_pages)} GitHub source files")
        all_pages.extend(gh_pages)

    if not all_pages:
        print("No pages found. Check your internet connection.")
        sys.exit(1)

    print(f"Building index for {len(all_pages)} total pages...")
    indexer = Indexer()
    indexer.index(all_pages)
    indexer.close()

    print("Done! Index built successfully.")
    print("Run `triton-mcp` to start the MCP server.")


if __name__ == "__main__":
    main()
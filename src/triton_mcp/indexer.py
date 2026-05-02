from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass

import chromadb
import tiktoken
from chromadb.config import Settings as ChromaSettings

from .config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_TOKENS,
    CHROMA_DIR,
    COLLECTION_NAME,
    SQLITE_PATH,
)
from .crawler import Page

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


@dataclass
class Chunk:
    chunk_id: str
    page_url: str
    page_title: str
    section: str
    chunk_index: int
    content: str
    code_blocks: str


def _build_chunks(pages: list[Page]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in pages:
        page_text = page.content
        if not page_text.strip():
            continue

        sections_text = "\n\n".join(
            f"## {s['heading']}\n{s['content']}" for s in page.sections
        ) if page.sections else ""

        combined = f"# {page.title}\n\n{sections_text}" if sections_text else page_text
        if len(combined.strip()) < len(page_text.strip()):
            combined = page_text

        text_chunks = _chunk_text(combined)
        code_joined = "\n\n---\n\n".join(page.code_blocks) if page.code_blocks else ""

        for i, chunk_content in enumerate(text_chunks):
            if i == 0 and code_joined:
                chunk_content = chunk_content + "\n\n### Code Examples\n\n" + code_joined[:2000]

            url_hash = hashlib.md5(page.url.encode()).hexdigest()[:8]
            chunk_id = f"{url_hash}_{i}"

            section_name = ""
            if page.sections:
                section_name = page.sections[0]["heading"]

            all_chunks.append(Chunk(
                chunk_id=chunk_id,
                page_url=page.url,
                page_title=page.title,
                section=section_name,
                chunk_index=i,
                content=chunk_content,
                code_blocks=code_joined,
            ))

    logger.info(f"Built {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks


class Indexer:
    def __init__(self) -> None:
        self.chroma_client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None
        self.sqlite_conn: sqlite3.Connection | None = None

    def _init_chroma(self) -> chromadb.Collection:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return self.collection

    def _init_sqlite(self) -> sqlite3.Connection:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
        self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                page_url TEXT NOT NULL,
                page_title TEXT NOT NULL,
                section TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                code_blocks TEXT
            )
        """)
        self.sqlite_conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                page_title,
                section,
                chunk_id UNINDEXED,
                tokenize='porter'
            )
        """)
        self.sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_url ON chunks(page_url)
        """)
        self.sqlite_conn.commit()
        return self.sqlite_conn

    def index(self, pages: list[Page]) -> None:
        chunks = _build_chunks(pages)
        if not chunks:
            logger.warning("No chunks to index")
            return

        collection = self._init_chroma()
        self._init_sqlite()
        assert self.sqlite_conn is not None

        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [
                {
                    "page_url": c.page_url,
                    "page_title": c.page_title[:500],
                    "section": c.section[:200],
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ]

            existing = collection.get(ids=ids, include=[])
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            collection.add(ids=ids, documents=documents, metadatas=metadatas)

            for c in batch:
                self.sqlite_conn.execute(
                    "INSERT OR REPLACE INTO chunks (chunk_id, page_url, page_title, section, chunk_index, content, code_blocks) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (c.chunk_id, c.page_url, c.page_title, c.section, c.chunk_index, c.content, c.code_blocks),
                )

            if start % 200 == 0:
                logger.info(f"Indexed {start + len(batch)}/{len(chunks)} chunks")

        self.sqlite_conn.execute("DELETE FROM chunks_fts")
        for c in chunks:
            self.sqlite_conn.execute(
                "INSERT INTO chunks_fts (content, page_title, section, chunk_id) VALUES (?, ?, ?, ?)",
                (c.content, c.page_title, c.section, c.chunk_id),
            )
        self.sqlite_conn.commit()
        logger.info(f"Indexing complete: {len(chunks)} chunks")

    def get_page_map(self) -> dict[str, str]:
        if not self.sqlite_conn:
            self._init_sqlite()
        assert self.sqlite_conn is not None
        rows = self.sqlite_conn.execute(
            "SELECT DISTINCT page_url, page_title FROM chunks ORDER BY page_url"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def close(self) -> None:
        if self.sqlite_conn:
            self.sqlite_conn.close()
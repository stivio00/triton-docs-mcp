from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import CHROMA_DIR, COLLECTION_NAME, SQLITE_PATH

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    chunk_id: str
    page_url: str
    page_title: str
    section: str
    content: str
    score: float


class SearchEngine:
    def __init__(self) -> None:
        self.chroma_client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None
        self.sqlite_conn: sqlite3.Connection | None = None

    def _ensure_connections(self) -> None:
        if self.chroma_client is None:
            self.chroma_client = chromadb.PersistentClient(
                path=str(CHROMA_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        if self.sqlite_conn is None:
            self.sqlite_conn = sqlite3.connect(str(SQLITE_PATH))
            self.sqlite_conn.execute("PRAGMA journal_mode=WAL")

    def _chunk_from_row(self, row: tuple) -> SearchResult:
        return SearchResult(
            chunk_id=row[0],
            page_url=row[1],
            page_title=row[2],
            section=row[3] or "",
            content=row[5],
            score=0.0,
        )

    def semantic_search(self, query: str, k: int = 5) -> list[SearchResult]:
        self._ensure_connections()
        assert self.collection is not None

        results = self.collection.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])

        if not results["ids"][0]:
            return []

        search_results: list[SearchResult] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            search_results.append(SearchResult(
                chunk_id=doc_id,
                page_url=meta["page_url"],
                page_title=meta["page_title"],
                section=meta.get("section", ""),
                content=results["documents"][0][i],
                score=1.0 - distance,
            ))

        return search_results

    def keyword_search(self, query: str, k: int = 5) -> list[SearchResult]:
        self._ensure_connections()
        assert self.sqlite_conn is not None

        escaped = query.replace('"', '""')
        fts_query = f'"{escaped}"'

        cursor = self.sqlite_conn.execute(
            """
            SELECT chunks.chunk_id, chunks.page_url, chunks.page_title, chunks.section, 
                   chunks.content, chunks.code_blocks, bm25(chunks_fts) as rank
            FROM chunks_fts
            JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank DESC
            LIMIT ?
            """,
            (fts_query, k),
        )
        rows = cursor.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            sr = SearchResult(
                chunk_id=row[0],
                page_url=row[1],
                page_title=row[2],
                section=row[3] or "",
                content=row[4],
                score=float(row[6]) if row[6] else 0.0,
            )
            results.append(sr)

        if not results:
            terms = query.split()
            if len(terms) > 1:
                fts_query = " OR ".join(f'"{t}"' for t in terms)
                cursor = self.sqlite_conn.execute(
                    """
                    SELECT chunks.chunk_id, chunks.page_url, chunks.page_title, chunks.section,
                           chunks.content, chunks.code_blocks, bm25(chunks_fts) as rank
                    FROM chunks_fts
                    JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank DESC
                    LIMIT ?
                    """,
                    (fts_query, k),
                )
                rows = cursor.fetchall()
                for row in rows:
                    results.append(SearchResult(
                        chunk_id=row[0],
                        page_url=row[1],
                        page_title=row[2],
                        section=row[3] or "",
                        content=row[4],
                        score=float(row[6]) if row[6] else 0.0,
                    ))

        return results

    def hybrid_search(self, query: str, k: int = 5) -> list[SearchResult]:
        semantic = self.semantic_search(query, k=k * 2)
        keyword = self.keyword_search(query, k=k * 2)

        scores: dict[str, float] = {}
        items: dict[str, SearchResult] = {}

        for rank, sr in enumerate(semantic):
            scores[sr.chunk_id] = scores.get(sr.chunk_id, 0) + 1.0 / (rank + 60)
            items[sr.chunk_id] = sr

        for rank, sr in enumerate(keyword):
            scores[sr.chunk_id] = scores.get(sr.chunk_id, 0) + 1.0 / (rank + 60)
            if sr.chunk_id not in items:
                items[sr.chunk_id] = sr

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [items[cid] for cid, _ in ranked[:k]]

    def get_page(self, url: str) -> list[SearchResult]:
        self._ensure_connections()
        assert self.sqlite_conn is not None

        cursor = self.sqlite_conn.execute(
            "SELECT chunk_id, page_url, page_title, section, chunk_index, content, code_blocks FROM chunks WHERE page_url = ? ORDER BY chunk_index",
            (url,),
        )
        rows = cursor.fetchall()
        return [SearchResult(
            chunk_id=row[0],
            page_url=row[1],
            page_title=row[2],
            section=row[3] or "",
            content=row[5],
            score=0.0,
        ) for row in rows]

    def list_pages(self) -> list[dict]:
        self._ensure_connections()
        assert self.sqlite_conn is not None

        cursor = self.sqlite_conn.execute(
            "SELECT DISTINCT page_url, page_title FROM chunks ORDER BY page_title"
        )
        return [{"url": row[0], "title": row[1]} for row in cursor.fetchall()]

    def close(self) -> None:
        if self.sqlite_conn:
            self.sqlite_conn.close()
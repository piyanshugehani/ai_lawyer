from __future__ import annotations
from typing import Dict, List, Any, Optional
from pathlib import Path
import os
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

# Real FAISS-backed retrieval
from batch_faiss_pipeline import (
    FaissShardManager,
    retrieve as faiss_retrieve,
    EMBED_DIM,
    OUTPUT_DIR,
    PDF_DIR,
    CHUNK_PREVIEW_CHARS,
    chunk_text_legal_aware,
)


def load_faiss_manager(court: Optional[str] = None) -> FaissShardManager:
    multi = os.getenv("MULTI_INDEX_BY_COURT", "0") == "1"
    if not multi:
        return FaissShardManager(OUTPUT_DIR, EMBED_DIM)
    # per-court directories when multi-index is enabled
    if court and court.upper() in {"SC", "HC"}:
        dir_name = f"faiss_shards_{court.upper()}"
    else:
        dir_name = "faiss_shards_other"
    out_dir = Path(__file__).with_name(dir_name)
    return FaissShardManager(out_dir, EMBED_DIM)


def _load_chunk_snippet(source_path: str, chunk_index: int, max_chars: int = 200) -> str:
    pdf_path = PDF_DIR / source_path
    if not pdf_path.exists():
        return ""
    try:
        if pdfplumber is None:
            return ""
        with pdfplumber.open(pdf_path) as pdf:
            pages_txt = []
            for p in pdf.pages:
                t = p.extract_text()
                if t:
                    pages_txt.append(t)
        full = "\n\n".join(pages_txt)
        chunks = chunk_text_legal_aware(full)
        if 0 <= chunk_index < len(chunks):
            txt = chunks[chunk_index]
            return txt[:max_chars] + ("..." if len(txt) > max_chars else "")
        return ""
    except Exception:
        return ""


def search_chunks(
    query: str,
    top_k: int = 5,
    court: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    doc_id_like: Optional[str] = None,
) -> List[Dict[str, Any]]:
    multi = os.getenv("MULTI_INDEX_BY_COURT", "0") == "1"
    filters_base: Dict[str, Any] = {}
    if year_min is not None:
        filters_base["year_min"] = int(year_min)
    if year_max is not None:
        filters_base["year_max"] = int(year_max)
    if doc_id_like:
        filters_base["doc_id_like"] = doc_id_like

    results: List[Dict[str, Any]] = []
    if multi and not court:
        # Query across SC, HC, and OTHER when court is unspecified
        for c in ["SC", "HC", None]:
            mgr = load_faiss_manager(court=c)
            filters = dict(filters_base)
            if c:
                filters["court"] = c
            try:
                hits = faiss_retrieve(query, mgr, top_k=top_k, rerank_with_gemini=True, filters=filters)
                results.extend(hits)
            except Exception:
                continue
        # Merge and take overall top_k by score (if present)
        def _score(h: Dict[str, Any]) -> float:
            try:
                return float(h.get("score", 0.0))
            except Exception:
                return 0.0
        results = sorted(results, key=_score, reverse=True)[:top_k]
    else:
        mgr = load_faiss_manager(court=court)
        filters = dict(filters_base)
        if court:
            filters["court"] = court
        results = faiss_retrieve(query, mgr, top_k=top_k, rerank_with_gemini=True, filters=filters)

    # Enrich with title/summary fields for compatibility with downstream code
    enriched: List[Dict[str, Any]] = []
    for h in results:
        snippet = _load_chunk_snippet(h.get("source_path", ""), int(h.get("chunk_index", -1)), max_chars=max(100, CHUNK_PREVIEW_CHARS * 10))
        enriched.append({
            **h,
            "title": h.get("doc_id") or h.get("source_path"),
            "summary": snippet or f"Chunk {h.get('chunk_index')} of {h.get('source_path')}",
        })
    return enriched


def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    # Thin wrapper for compatibility with older callers
    return search_chunks(query, top_k=k)

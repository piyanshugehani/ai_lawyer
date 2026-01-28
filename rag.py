from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# External Libraries
from pinecone import Pinecone
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
except Exception:
    pass

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Separate indexes for Supreme Court (SCC), High Court (HCC) and Bare Acts
INDEX_NAME_SC = "legal-judgments-index"
INDEX_NAME_HC = "high-court-judgments-index"
INDEX_NAME_BARE = os.getenv("BARE_ACTS_INDEX_NAME", "legal-bare-acts-index")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or GEMINI_API_KEY.")

# ---------------------------------------------------------
# INITIALIZE CLIENTS
# ---------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Individual indexes for SCC, HCC and Bare Acts
index_sc = pc.Index(INDEX_NAME_SC)
index_hc = pc.Index(INDEX_NAME_HC)
try:
    index_bare = pc.Index(INDEX_NAME_BARE)
except Exception as e:
    # Bare Acts index is optional; log and continue gracefully if missing
    logger.warning(f"Bare Acts index initialisation failed: {e}")
    index_bare = None

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_query_embedding(text: str) -> Optional[List[float]]:
    """
    Generates an embedding for the query using the current 'gemini-embedding-001' model.
    CRITICAL: We force 'outputDimensionality': 768 to match Pinecone's free tier index.
    """
    try:
        # URL for the new stable model (v1beta is required for this model as of 2026)
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-embedding-001:embedContent"
        )

        payload = {
            "model": "models/gemini-embedding-001",
            "content": {
                "parts": [{"text": text}]
            },
            # FORCE 768 DIMENSIONS (Default is 3072, which breaks Pinecone free tier)
            "outputDimensionality": 768
        }

        params = {"key": GEMINI_API_KEY}
        
        resp = requests.post(url, params=params, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse the response
        embedding = data.get("embedding") or {}
        values = embedding.get("values")
        
        if values:
            return values
        else:
            logger.error(f"No values found in response: {data}")
            return None

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        if 'resp' in locals():
            logger.error(f"Response content: {resp.text}")
        return None

def build_pinecone_filter(
    court: Optional[str] = None, 
    year_min: Optional[int] = None, 
    year_max: Optional[int] = None,
    doc_id_like: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs the Pinecone metadata filter.
    """
    filters = {}

    # 1. Filter by Year
    if year_min is not None or year_max is not None:
        filters["year"] = {}
        if year_min is not None:
            filters["year"]["$gte"] = int(year_min)
        if year_max is not None:
            filters["year"]["$lte"] = int(year_max)

    # 2. Filter by Filename
    if doc_id_like:
        filters["filename"] = doc_id_like

    return filters


def build_bare_acts_filter(
    act_id_like: Optional[str] = None,
    act_title_like: Optional[str] = None,
    section_number_like: Optional[str] = None,
) -> Dict[str, Any]:
    """Constructs filter for Bare Acts index."""
    filt: Dict[str, Any] = {}
    if act_id_like:
        filt["act_id"] = act_id_like
    if act_title_like:
        filt["act_title"] = act_title_like
    if section_number_like:
        filt["section_number"] = section_number_like
    return filt

# ---------------------------------------------------------
# MAIN SEARCH FUNCTION
# ---------------------------------------------------------

def search_chunks(
    query: str,
    top_k: int = 5,
    court: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    doc_id_like: Optional[str] = None,
    keyword_must_contain: Optional[str] = None,
    prefer_high_court: bool = False,
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant chunks using semantic search + metadata filtering.
    """
    
    # 1. Generate Vector
    vector = get_query_embedding(query)
    if not vector:
        logger.error("Could not generate embedding for query.")
        return []

    # 2. Build Filter
    meta_filter = build_pinecone_filter(court, year_min, year_max, doc_id_like)

    # 3. Decide which indexes to query
    court_norm = (court or "").lower().strip()
    indexes_to_query: List[Tuple[str, Any]]
    
    if court_norm in ["hc", "hcc", "high", "high court"]:
        indexes_to_query = [("hcc", index_hc)]
    elif court_norm in ["both", "all"]:
        indexes_to_query = [("scc", index_sc), ("hcc", index_hc)]
    else:
        # Default to Supreme Court
        indexes_to_query = [("scc", index_sc)]

    # 4. Query Pinecone
    fetch_k = top_k * 3 if keyword_must_contain else top_k
    all_matches: List[Tuple[str, Any]] = []

    for label, idx in indexes_to_query:
        try:
            results = idx.query(
                vector=vector,
                top_k=fetch_k,
                include_metadata=True,
                filter=meta_filter if meta_filter else None,
            )
            for match in results.matches:
                all_matches.append((label, match))
        except Exception as e:
            logger.error(f"Pinecone query failed for index '{label}': {e}")
            continue

    if not all_matches:
        return []

    # 5. Sort matches
    def _score_with_boost(label: str, base_score: float) -> float:
        if prefer_high_court and label == "hcc":
            return base_score + 0.05
        return base_score

    all_matches.sort(key=lambda x: _score_with_boost(x[0], x[1].score), reverse=True)

    # 6. Parse Results
    enriched_results: List[Dict[str, Any]] = []

    for label, match in all_matches:
        md = match.metadata or {}
        text_content = md.get("text", "")
        
        # Keyword check
        if keyword_must_contain and keyword_must_contain.lower() not in text_content.lower():
            continue

        enriched_results.append({
            "doc_id": md.get("filename", "Unknown"),
            "source_path": md.get("filename", "Unknown"),
            "chunk_index": int(md.get("chunk_index", 0)),
            "score": match.score,
            "title": f"{md.get('filename')} (Page {md.get('page')})",
            "year": int(md.get("year", 0)),
            "category": md.get("category") or label,
            "court": md.get("court"),
            "court_level": md.get("court_level"),
            "page": int(md.get("page", 0)),
            "summary": text_content,
            "text": text_content,
        })

    return enriched_results[:top_k]


def search_bare_acts(
    query: str,
    top_k: int = 5,
    act_id_like: Optional[str] = None,
    act_title_like: Optional[str] = None,
    section_number_like: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic search over the Bare Acts index."""

    if index_bare is None:
        return []

    vector = get_query_embedding(query)
    if not vector:
        return []

    meta_filter = build_bare_acts_filter(
        act_id_like=act_id_like,
        act_title_like=act_title_like,
        section_number_like=section_number_like,
    )

    try:
        results = index_bare.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=meta_filter or None,
        )
    except Exception as e:
        logger.error(f"Pinecone query failed for Bare Acts index: {e}")
        return []

    enriched: List[Dict[str, Any]] = []

    for match in results.matches:
        md = match.metadata or {}
        text_content = md.get("text", "")
        
        enriched.append({
            "doc_id": md.get("act_id", "Unknown Act ID"),
            "source_path": md.get("act_title", "Bare Act"),
            "chunk_index": int(md.get("chunk_index", 0)),
            "score": match.score,
            "title": md.get("act_title", "Act"),
            "year": int(md.get("enactment_year", 0)),
            "category": md.get("doc_type", "bare_act"),
            "court": md.get("jurisdiction", "India"),
            "court_level": "Bare Act",
            "page": 0,
            "summary": text_content,
            "text": text_content,
        })

    return enriched

def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Legacy wrapper."""
    return search_chunks(query, top_k=k)

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    print("--- Testing Retrieval with gemini-embedding-001 (768 dim) ---")
    
    test_query = "swings slides fun fliers physical exercise"
    print(f"\nSearch: '{test_query}'")
    results = search_chunks(test_query, top_k=3)
    
    if not results:
        print("No results found. NOTE: If you just switched models, you MUST re-ingest your documents.")
        print("The old vectors (from text-embedding-004) are incompatible with the new model.")
    
    for r in results:
        print("-" * 40)
        print(f"Found: {r['doc_id']}")
        print(f"Page: {r['page']} | Year: {r['year']} | Score: {r['score']:.4f}")
        print(f"Snippet: {r['text'][:150]}...")
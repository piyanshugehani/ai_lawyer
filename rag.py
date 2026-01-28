from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# External Libraries
from pinecone import Pinecone
from google import genai

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

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_query_embedding(text: str) -> Optional[List[float]]:
    """
    Generates an embedding for the query using Gemini.
    """
    try:
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"}
        )
        if hasattr(resp, 'embeddings') and len(resp.embeddings) > 0:
            return resp.embeddings[0].values
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
    return None

def build_pinecone_filter(
    court: Optional[str] = None, 
    year_min: Optional[int] = None, 
    year_max: Optional[int] = None,
    doc_id_like: Optional[str] = None
) -> Dict[str, Any]:
    """
    Constructs the Pinecone metadata filter based on common fields:
    - year (int)
    - filename (str)

    NOTE: Court/SCC/HCC routing is handled at the index level, not via metadata
    filters, because Supreme Court and High Court are stored in separate
    Pinecone indexes.
    """
    filters = {}

    # 1. Filter by Year -> Maps to 'year' field (common to both indexes)
    if year_min is not None or year_max is not None:
        filters["year"] = {}
        if year_min is not None:
            filters["year"]["$gte"] = int(year_min)
        if year_max is not None:
            filters["year"]["$lte"] = int(year_max)

    # 2. Filter by Filename -> Maps to 'filename' field
    if doc_id_like:
        # Exact match required for basic Pinecone metadata
        filters["filename"] = doc_id_like

    return filters


def build_bare_acts_filter(
    act_id_like: Optional[str] = None,
    act_title_like: Optional[str] = None,
    section_number_like: Optional[str] = None,
) -> Dict[str, Any]:
    """Constructs a simple metadata filter for the Bare Acts index.

    All fields are optional and match the metadata schema you described:
    - act_id (e.g. "ACT NO. 4 OF 1975")
    - act_title (e.g. "THE TOBACCO BOARD ACT, 1975")
    - section_number (e.g. "definitions", "5", "5A")

    For now we use exact-equality filters, which work well when the
    query is pre-parsed into these fields. The main entrypoint from the
    app uses purely semantic search, so this filter is currently unused
    but kept ready for future refinement.
    """
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
    Retrieves relevant chunks using your specific Metadata structure.
    """
    
    # 1. Generate Vector
    vector = get_query_embedding(query)
    if not vector:
        return []

    # 2. Build Filter (shared across indexes)
    meta_filter = build_pinecone_filter(court, year_min, year_max, doc_id_like)

    # 3. Decide which indexes to query based on `court`
    court_norm = (court or "").lower().strip()

    # Default behaviour: Supreme Court only
    indexes_to_query: List[Tuple[str, Any]]
    if court_norm in ["hc", "hcc", "high", "high court"]:
        indexes_to_query = [("hcc", index_hc)]
    elif court_norm in ["both", "all"]:
        indexes_to_query = [("scc", index_sc), ("hcc", index_hc)]
    else:
        # Includes "sc", "scc", "supreme", "supreme court", or empty
        indexes_to_query = [("scc", index_sc)]

    # 4. Query Pinecone for each selected index
    # Fetch extra if we plan to do keyword filtering to ensure we return enough results
    fetch_k = top_k * 3 if keyword_must_contain else top_k

    all_matches: List[Tuple[str, Any]] = []  # (label, match)

    for label, idx in indexes_to_query:
        try:
            results = idx.query(
                vector=vector,
                top_k=fetch_k,
                include_metadata=True,  # Critical: We need your 'text' field
                filter=meta_filter if meta_filter else None,
            )
        except Exception as e:
            logger.error(f"Pinecone query failed for index '{label}': {e}")
            continue

        for match in results.matches:
            all_matches.append((label, match))

    if not all_matches:
        return []

    # 5. Sort all matches by score (descending), with optional minor boost
    # for High Court chunks when explicitly preferred (procedural queries).
    def _score_with_boost(label: str, base_score: float) -> float:
        if prefer_high_court and label == "hcc":
            return base_score + 0.05
        return base_score

    all_matches.sort(key=lambda x: _score_with_boost(x[0], x[1].score), reverse=True)

    # 6. Parse Results based on your Metadata Structure
    enriched_results: List[Dict[str, Any]] = []

    for label, match in all_matches:
        md = match.metadata or {}

        # --- MAPPING YOUR SPECIFIC FIELDS ---
        text_content = md.get("text", "")            # Field: text
        filename = md.get("filename", "Unknown")     # Field: filename
        page = int(md.get("page", 0))                # Field: page
        year = int(md.get("year", 0))                # Field: year
        # Prefer explicit category, then court_level, finally index label
        court_level_md = md.get("court_level")
        category = md.get("category") or court_level_md or label
        court_name_md = md.get("court")
        chunk_idx = int(md.get("chunk_index", 0))    # Field: chunk_index
        # ------------------------------------

        # Keyword Enforcement (Post-Retrieval Filter)
        if keyword_must_contain and keyword_must_contain.lower() not in text_content.lower():
            continue

        enriched_results.append({
            "doc_id": filename,
            "source_path": filename,
            "chunk_index": chunk_idx,
            "score": match.score,
            "title": f"{filename} (Page {page})",
            "year": year,
            "category": category,
            "court": court_name_md,
            "court_level": court_level_md,
            "page": page,
            "summary": text_content,  # Used by downstream LLM
            "text": text_content,     # Raw text access
        })

    # Return top_k combined across indexes
    return enriched_results[:top_k]


def search_bare_acts(
    query: str,
    top_k: int = 5,
    act_id_like: Optional[str] = None,
    act_title_like: Optional[str] = None,
    section_number_like: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic search over the Bare Acts index.

    Metadata schema (per your description):
    - act_id:          "ACT NO. 4 OF 1975"
    - act_title:       "THE TOBACCO BOARD ACT, 1975"
    - doc_type:        e.g. "bare_act_definition"
    - enactment_year:  1975
    - jurisdiction:    "India"
    - section_number:  e.g. "definitions", "5", "5A"
    - source:          "legislative_text"
    - text:            full statutory / preamble text
    """

    if index_bare is None:
        # Bare Acts index not configured; fail soft.
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
        act_id = md.get("act_id") or "Unknown Act ID"
        act_title = md.get("act_title") or "Bare Act Provision"
        section_number = md.get("section_number") or ""
        enactment_year_raw = md.get("enactment_year")
        try:
            enactment_year = int(enactment_year_raw) if enactment_year_raw is not None else 0
        except (TypeError, ValueError):
            enactment_year = 0

        doc_type = md.get("doc_type") or "bare_act"
        jurisdiction = md.get("jurisdiction") or "India"

        title_suffix = f" - Section {section_number}" if section_number else ""

        enriched.append({
            "doc_id": act_id,
            "source_path": act_title,
            "chunk_index": int(md.get("chunk_index", 0)),
            "score": match.score,
            "title": f"{act_title}{title_suffix}",
            "year": enactment_year,
            "category": doc_type,
            "court": jurisdiction,
            "court_level": "Bare Act",
            "page": 0,
            "summary": text_content,
            "text": text_content,
            "act_id": act_id,
            "act_title": act_title,
            "section_number": section_number,
            "doc_type": doc_type,
            "jurisdiction": jurisdiction,
        })

    return enriched

def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Legacy wrapper."""
    return search_chunks(query, top_k=k)

# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------
if __name__ == "__main__":
    # Test based on your specific sample data
    print("--- Testing Retrieval with your Metadata ---")
    
    # 1. Try to find the exact chunk you showed me (Toy vs Exercise equipment)
    print("\nSearch: 'swings slides fun fliers'")
    results = search_chunks("swings slides fun fliers physical exercise", top_k=3)
    
    for r in results:
        print(f"Found: {r['doc_id']}")
        print(f"Page: {r['page']} | Year: {r['year']} | Cat: {r['category']}")
        print(f"Snippet: {r['text'][:100]}...")
from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# External Libraries
try:
    from pinecone import Pinecone
    from google import genai
    from google.genai import types
    from dotenv import load_dotenv
except ImportError:
    raise ImportError("Missing libraries. Run: pip install -U google-genai pinecone-client python-dotenv")

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Pinecone Config
INDEX_NAME = "legal-judgments-index"
BARE_ACTS_INDEX_NAME = os.getenv("BARE_ACTS_INDEX_NAME", "legal-bare-acts-index")

# Namespace configuration
# Supreme Court PDFs live in two namespaces; High Court PDFs live in one.
SC_NAMESPACES = ["supreme_court", "high_court"]
HC_NAMESPACES = ["supreme_court", "high_court","HC"]

# Model Config
EMBEDDING_MODEL = "models/gemini-embedding-001"

# Retrieval tuning
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.25"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing PINECONE_API_KEY or GEMINI_API_KEY in .env file")

# ---------------------------------------------------------
# INITIALIZE CLIENTS
# ---------------------------------------------------------
client_genai = genai.Client(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

try:
    index_judgments = pc.Index(INDEX_NAME)
except Exception as e:
    logger.error(f"❌ Failed to connect to Judgments Index '{INDEX_NAME}': {e}")
    index_judgments = None

try:
    index_bare = pc.Index(BARE_ACTS_INDEX_NAME)
except Exception as e:
    logger.warning(f"Bare Acts index '{BARE_ACTS_INDEX_NAME}' not found. Skipping.")
    index_bare = None

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_query_embedding(text: str) -> Optional[List[float]]:
    """Generates an embedding using the currently configured EMBEDDING_MODEL."""
    try:
        # We wrap 'contents=[text]' to ensure correct SDK behavior
        response = client_genai.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        if response.embeddings and len(response.embeddings) > 0:
            return response.embeddings[0].values
        return None
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None

def build_pinecone_filter(
    year_min: Optional[int] = None, 
    year_max: Optional[int] = None,
    doc_id_like: Optional[str] = None
) -> Dict[str, Any]:
    filters = {}
    if year_min or year_max:
        filters["year"] = {}
        if year_min: filters["year"]["$gte"] = int(year_min)
        if year_max: filters["year"]["$lte"] = int(year_max)
    if doc_id_like:
        filters["filename"] = doc_id_like
    return filters


def build_bare_acts_filter(
    act_id_like: Optional[str] = None,
    act_title_like: Optional[str] = None,
    section_number_like: Optional[str] = None,
) -> Dict[str, Any]:
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
    
    if not index_judgments:
        logger.error("Index not initialized.")
        return []

    # 1. Generate Vector
    vector = get_query_embedding(query)
    if not vector:
        return []

    # 2. Build Filter
    meta_filter = build_pinecone_filter(year_min, year_max, doc_id_like)

    # 3. Determine Namespaces based on court routing
    #    "scc"  -> only Supreme Court namespaces
    #    "hc"   -> only High Court namespace
    #    "both"/None/other -> all namespaces
    court = (court or "both").lower()
    if court == "scc":
        target_namespaces = SC_NAMESPACES
    elif court == "hc":
        target_namespaces = HC_NAMESPACES
    else:
        target_namespaces = SC_NAMESPACES + HC_NAMESPACES

    # 4. Query Pinecone (with safe fallbacks)
    all_matches = []
    for ns in target_namespaces:
        try:
            # First attempt: with metadata filters (year / filename) if any
            results = index_judgments.query(
                vector=vector,
                namespace=ns,
                top_k=top_k,
                include_metadata=True,
                filter=meta_filter if meta_filter else None,
            )
            ns_matches = list(results.matches or [])

            # If filters were applied and nothing came back, retry once without filters
            if meta_filter and not ns_matches:
                logger.info(
                    "RAG: namespace=%s had no matches with filter=%s, retrying without filter",
                    ns,
                    meta_filter,
                )
                results = index_judgments.query(
                    vector=vector,
                    namespace=ns,
                    top_k=top_k,
                    include_metadata=True,
                    filter=None,
                )
                ns_matches = list(results.matches or [])

            for match in ns_matches:
                all_matches.append((ns, match))
        except Exception as e:
            logger.error(f"RAG query failed for namespace={ns}: {e}")
            continue

    if not all_matches:
        logger.info("RAG: No matches from Pinecone across any namespace (even without filters).")
        return []

    # Basic debug stats for raw matches before any score filtering
    raw_scores = [m.score or 0.0 for _, m in all_matches]
    if raw_scores:
        logger.info(
            "RAG raw matches: count=%d top=%.3f min=%.3f avg=%.3f threshold=%.3f",
            len(raw_scores),
            max(raw_scores),
            min(raw_scores),
            sum(raw_scores) / len(raw_scores),
            MIN_SIMILARITY_SCORE,
        )

    # 5. Sort by Score (optionally give a tiny boost to High Court)
    def _sort_key(item):
        ns, match = item
        score = match.score or 0.0
        if prefer_high_court and ns in HC_NAMESPACES:
            score += 0.05
        return score

    all_matches.sort(key=_sort_key, reverse=True)

    # 6. Format Output
    enriched_results: List[Dict[str, Any]] = []
    seen_ids = set()
    below_threshold_count = 0

    for ns, match in all_matches:
        md = match.metadata or {}
        doc_id = md.get("filename", "Unknown")

        # Deduplication (optional)
        if doc_id in seen_ids and len(enriched_results) > 0:
            continue
        seen_ids.add(doc_id)

        text_content = md.get("text", "")

        # Drop very low-similarity chunks so we don't hallucinate
        score_val = match.score or 0.0
        if score_val < MIN_SIMILARITY_SCORE:
            below_threshold_count += 1
            continue

        # Optional keyword filter at chunk level
        if keyword_must_contain and keyword_must_contain.lower() not in text_content.lower():
            continue

        enriched_results.append({
            "doc_id": doc_id,
            "source_path": doc_id,
            "chunk_index": int(md.get("chunk_index", 0)),
            "score": score_val,
            "title": doc_id,
            "year": int(md.get("year", 0)),
            "category": "High Court" if ns in HC_NAMESPACES else "Supreme Court",
            "court": md.get("court", ns),
            "page": int(md.get("page", 0)),
            "summary": text_content,
            "text": text_content,
        })

        if len(enriched_results) >= top_k:
            break

    # If Pinecone returned matches but all were under the similarity threshold,
    # fall back to the best top_k matches instead of returning an empty list.
    if not enriched_results and all_matches:
        logger.warning(
            "RAG: %d matches were below similarity threshold %.3f; "
            "falling back to unfiltered top_k results.",
            below_threshold_count,
            MIN_SIMILARITY_SCORE,
        )

        enriched_results = []
        seen_ids.clear()
        for ns, match in all_matches:
            md = match.metadata or {}
            doc_id = md.get("filename", "Unknown")

            if doc_id in seen_ids and len(enriched_results) > 0:
                continue
            seen_ids.add(doc_id)

            text_content = md.get("text", "")

            if keyword_must_contain and keyword_must_contain.lower() not in text_content.lower():
                continue

            enriched_results.append({
                "doc_id": doc_id,
                "source_path": doc_id,
                "chunk_index": int(md.get("chunk_index", 0)),
                "score": match.score or 0.0,
                "title": doc_id,
                "year": int(md.get("year", 0)),
                "category": "High Court" if ns in HC_NAMESPACES else "Supreme Court",
                "court": md.get("court", ns),
                "page": int(md.get("page", 0)),
                "summary": text_content,
                "text": text_content,
            })

            if len(enriched_results) >= top_k:
                break

    return enriched_results


def search_bare_acts(
    query: str,
    top_k: int = 5,
    act_id_like: Optional[str] = None,
    act_title_like: Optional[str] = None,
    section_number_like: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if not index_bare:
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
            "title": f"{md.get('act_title')} - Section {md.get('section_number')}",
            "year": int(md.get("enactment_year", 0)),
            "category": "Bare Act",
            "court": "Legislation",
            "page": 0,
            "summary": text_content,
            "text": text_content,
        })

    return enriched

def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    return search_chunks(query, top_k=k)

# ---------------------------------------------------------
# TEST BLOCK
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Test...")
    test_query = "Ambuja Cement sales tax exemption Himachal Pradesh"
    
    results = search_chunks(test_query, top_k=3, court="both")
    
    if results:
        print(f"\n✅ Success! Found {len(results)} matches:\n")
        for r in results:
            excerpt = (r.get("text") or "")[:150].replace("\n", " ")
            print(f"📄 {r['doc_id']} (Year: {r['year']})")
            print(f"   Excerpt: {excerpt}...\n")
    else:
        print("\n❌ No results found (Check Index Stats).")
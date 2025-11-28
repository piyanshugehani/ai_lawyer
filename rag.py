from __future__ import annotations
import os
import logging
from typing import Dict, List, Any, Optional
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
INDEX_NAME = "legal-judgments-index"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or GEMINI_API_KEY.")

# ---------------------------------------------------------
# INITIALIZE CLIENTS
# ---------------------------------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
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
    Constructs the Pinecone metadata filter based on your fields:
    - category ("scc" or "hcc")
    - year (int)
    - filename (str)
    """
    filters = {}

    # 1. Filter by Court -> Maps to 'category' field
    if court:
        c_val = court.lower().strip()
        # Map common user terms to your specific metadata values
        if c_val in ["sc", "scc", "supreme", "supreme court"]:
            filters["category"] = "scc"
        elif c_val in ["hc", "hcc", "high", "high court"]:
            filters["category"] = "hcc"

    # 2. Filter by Year -> Maps to 'year' field
    if year_min is not None or year_max is not None:
        filters["year"] = {}
        if year_min is not None:
            filters["year"]["$gte"] = int(year_min)
        if year_max is not None:
            filters["year"]["$lte"] = int(year_max)

    # 3. Filter by Filename -> Maps to 'filename' field
    if doc_id_like:
        # Exact match required for basic Pinecone metadata
        filters["filename"] = doc_id_like

    return filters

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
    keyword_must_contain: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves relevant chunks using your specific Metadata structure.
    """
    
    # 1. Generate Vector
    vector = get_query_embedding(query)
    if not vector:
        return []

    # 2. Build Filter
    meta_filter = build_pinecone_filter(court, year_min, year_max, doc_id_like)

    # 3. Query Pinecone
    # Fetch extra if we plan to do keyword filtering to ensure we return enough results
    fetch_k = top_k * 3 if keyword_must_contain else top_k

    try:
        results = index.query(
            vector=vector,
            top_k=fetch_k,
            include_metadata=True, # Critical: We need your 'text' field
            filter=meta_filter if meta_filter else None
        )
    except Exception as e:
        logger.error(f"Pinecone query failed: {e}")
        return []

    # 4. Parse Results based on your Metadata Structure
    enriched_results = []
    
    for match in results.matches:
        md = match.metadata or {}
        
        # --- MAPPING YOUR SPECIFIC FIELDS ---
        text_content = md.get("text", "")          # Field: text
        filename = md.get("filename", "Unknown")   # Field: filename
        page = int(md.get("page", 0))              # Field: page
        year = int(md.get("year", 0))              # Field: year
        category = md.get("category", "")          # Field: category
        chunk_idx = int(md.get("chunk_index", 0))  # Field: chunk_index
        # ------------------------------------

        # Keyword Enforcement (Post-Retrieval Filter)
        if keyword_must_contain:
            if keyword_must_contain.lower() not in text_content.lower():
                continue 

        enriched_results.append({
            "doc_id": filename,
            "source_path": filename,
            "chunk_index": chunk_idx,
            "score": match.score,
            "title": f"{filename} (Page {page})",
            "year": year,
            "category": category,
            "page": page,
            "summary": text_content, # Used by downstream LLM
            "text": text_content     # Raw text access
        })

    # Return top_k
    return enriched_results[:top_k]

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
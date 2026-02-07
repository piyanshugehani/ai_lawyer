from __future__ import annotations
import argparse
import textwrap
from typing import List, Dict, Any

import rag


def format_result(doc: Dict[str, Any], index: int) -> str:
    """Pretty-print a single retrieved document with key metadata."""
    title = str(doc.get("title") or doc.get("doc_id") or "<no title>")
    score = float(doc.get("score") or 0.0)
    year = doc.get("year")
    court = doc.get("court") or doc.get("category") or ""
    page = doc.get("page")

    header_parts = [f"#{index}", f"score={score:.3f}"]
    if year:
        header_parts.append(f"year={year}")
    if court:
        header_parts.append(f"court={court}")
    if isinstance(page, int) and page > 0:
        header_parts.append(f"page={page}")

    header = " | ".join(header_parts)

    text = (doc.get("text") or doc.get("summary") or "").replace("\n", " ")
    excerpt = textwrap.shorten(text, width=400, placeholder=" …") if text else "<no text>"

    return f"{header}\nTitle: {title}\nExcerpt: {excerpt}\n"


def run_query(query: str, top_k: int = 5, court: str = "both") -> None:
    """Run a single RAG retrieval query and print top-k results.

    court options (as used in rag.search_chunks):
      - "scc"  : Supreme Court namespaces only (supreme_court, high_court)
      - "hc"   : High Court namespace only (HC)
      - "both" : All namespaces together
    """
    print("=" * 80)
    print(f"RAG test query: {query!r}")
    print(f"top_k={top_k}, court={court}")
    print("=" * 80)

    results: List[Dict[str, Any]] = rag.search_chunks(
        query=query,
        top_k=top_k,
        court=court,
        prefer_high_court=False,
    )

    if not results:
        print("No documents retrieved. Check index connectivity, filters, or similarity threshold.")
        return

    print(f"\nRetrieved {len(results)} document(s):\n")
    for idx, doc in enumerate(results, start=1):
        print(format_result(doc, idx))
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG retrieval test for legal-judgments-index.")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Natural language legal query to test retrieval with. If omitted, you'll be prompted.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top documents to return (default: 5).")
    parser.add_argument(
        "--court",
        type=str,
        default="both",
        choices=["scc", "hc", "both"],
        help=(
            "Court routing for namespaces: 'scc' (Supreme Court only), "
            "'hc' (High Court only), 'both' (all namespaces). Default: both"
        ),
    )

    args = parser.parse_args()

    query = args.query
    if not query:
        try:
            query = input("Enter legal query to test retrieval with: ").strip()
        except EOFError:
            query = ""

    if not query:
        print("No query provided. Exiting.")
        raise SystemExit(1)

    run_query(query, top_k=args.top_k, court=args.court)

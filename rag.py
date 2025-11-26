from __future__ import annotations
import math
from typing import Dict, List, Tuple

# Fake vector databases
SUPREME_CASES = [
    {
        "id": f"SC{i}",
        "title": title,
        "summary": summary,
        "text": summary + " This is a landmark Supreme Court decision used here as dummy data.",
        "source": "supreme",
    }
    for i, (title, summary) in enumerate(
        [
            ("Kesavananda Bharati v. State of Kerala (1973)", "Basic structure doctrine under the Constitution."),
            ("Maneka Gandhi v. Union of India (1978)", "Due process and Article 21 expanded."),
            ("Vishaka v. State of Rajasthan (1997)", "Guidelines on sexual harassment at workplace."),
            ("Shreya Singhal v. Union of India (2015)", "Struck down Section 66A IT Act for violating free speech."),
            ("Navtej Singh Johar v. Union of India (2018)", "Decriminalized consensual same-sex relations."),
        ]
    )
]

HIGH_COURT_CASES = [
    {
        "id": f"HC{i}",
        "title": title,
        "summary": summary,
        "text": summary + " This is a dummy High Court decision.",
        "source": "high",
    }
    for i, (title, summary) in enumerate(
        [
            ("XYZ v. State of Delhi (2020)", "Bail jurisprudence in economic offences."),
            ("ABC v. State of Karnataka (2019)", "Contract specific performance nuances."),
            ("LMN v. PQR (2017)", "Property dispute on adverse possession."),
            ("EFG v. UVW (2018)", "Family court decree on maintenance."),
            ("RST v. OPQ (2021)", "GST refund and limitation period."),
        ]
    )
]


# Hash-based fake embedding
_DEF_DIM = 16


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


def embed(text: str, dim: int = _DEF_DIM) -> List[float]:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        idx = (hash(tok) % dim + dim) % dim
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def retrieve(query: str, k: int = 3) -> List[Dict]:
    """Return top-k fake cases from both DBs based on cosine similarity."""
    qv = embed(query)
    all_cases = SUPREME_CASES + HIGH_COURT_CASES
    scored: List[Tuple[float, Dict]] = []
    for c in all_cases:
        cv = embed(c["title"] + " " + c["summary"])
        sim = cosine(qv, cv)
        scored.append((sim, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, c in scored[: max(1, k)]:
        item = dict(c)
        item["score"] = round(sim, 3)
        results.append(item)
    return results

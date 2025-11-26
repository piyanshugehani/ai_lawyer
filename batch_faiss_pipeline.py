from __future__ import annotations

"""
Batch ingests text PDFs into FAISS vector shards.
Stores metadata in a SQLite DB for filtering and provenance.
Resumable: checkpoints processed files, shards persist to disk.
Retrieval example at bottom.

Requirements:
pip install pdfplumber langchain faiss-cpu tiktoken sqlite-utils python-dotenv numpy
(Replace faiss-cpu with faiss-gpu if you have GPU-enabled FAISS)
"""

import os
import re
import json
import time
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Iterable
from dataclasses import dataclass

import numpy as np
import pdfplumber
import faiss
from dotenv import load_dotenv
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # defer error until embedding call
    genai = None  # type: ignore

try:  # prefer the standalone splitter package if available
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
    except Exception:
        # Minimal fallback splitter if LangChain not installed
        class RecursiveCharacterTextSplitter:  # type: ignore
            def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200, separators=None):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", ". "]

            def split_text(self, text: str):
                if not text:
                    return []
                # coarse split by first separator available
                parts = [text]
                for sep in self.separators:
                    new_parts = []
                    for p in parts:
                        new_parts.extend(p.split(sep))
                    parts = new_parts
                    if len(parts) > 1:
                        break
                # assemble chunks with overlap
                chunks: List[str] = []
                buf = []
                char_count = 0
                for piece in parts:
                    piece = piece.strip()
                    if not piece:
                        continue
                    if char_count + len(piece) > self.chunk_size:
                        combined = " ".join(buf).strip()
                        if combined:
                            chunks.append(combined)
                        # start new buffer with overlap from end of previous
                        if combined and self.chunk_overlap > 0:
                            overlap_slice = combined[-self.chunk_overlap :]
                            buf = [overlap_slice, piece]
                            char_count = len(overlap_slice) + len(piece)
                        else:
                            buf = [piece]
                            char_count = len(piece)
                    else:
                        buf.append(piece)
                        char_count += len(piece)
                final = " ".join(buf).strip()
                if final:
                    chunks.append(final)
                return chunks


# -------------
# CONFIG
# -------------
# Use the repository’s provided PDF folder
PDF_DIR = Path(__file__).with_name("supreme-court-pdfs-05-06")
OUTPUT_DIR = Path(__file__).with_name("faiss_shards")
METADB_PATH = Path(__file__).with_name("metadata.db")
PROCESSED_LOG = Path(__file__).with_name("processed.json")

BATCH_SIZE = 256                    # larger embedding batches reduce overhead
PDF_BATCH = 100                     # process more PDFs per loop to reduce orchestration cost
CHUNK_SIZE = 1000                   # slightly smaller chunks to reduce tokens per call
CHUNK_OVERLAP = 100                 # smaller overlap reduces total chunks

# Gemini text-embedding-004 returns 768-dim vectors
EMBED_DIM = 768
# Allow shard size override via env
SHARD_MAX_VECTORS = int(os.getenv("SHARD_MAX_VECTORS", "2000000"))
INDEX_FACTORY = "HNSW16"            # faster to build, still good recall
USE_ID_MAPPING = True

LOG = logging.getLogger("batch_faiss")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Printing controls
PRINT_PDF_TEXT = True               # print extracted text from each PDF
PRINT_MAX_CHARS = 30              # truncate long prints to this many characters
PRINT_CHUNK_PREVIEWS = True         # show short previews of first few chunks
CHUNK_PREVIEW_COUNT = 2
CHUNK_PREVIEW_CHARS = 10

# Embedding concurrency
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "8"))  # threads for parallel embed calls

# Shard rotation policy
# Set FORCE_NEW_SHARD=1 to start a fresh shard for each mini-batch add
FORCE_NEW_SHARD = os.getenv("FORCE_NEW_SHARD", "0") == "1"

# Full rebuild control (set RESET=1 to clear shards + metadata and re-ingest from start)
RESET_ALL = os.getenv("RESET", "0") == "1"

# FAISS GPU mode: "cpu" (default), "gpu_sharded", or "gpu_replicated"
FAISS_GPU_MODE = os.getenv("FAISS_GPU_MODE", "cpu").lower()

# Multi-index by court: when enabled, builds separate output dirs per court (SC/HC/OTHER)
MULTI_INDEX_BY_COURT = os.getenv("MULTI_INDEX_BY_COURT", "0") == "1"

# Rotate shards after N PDFs (per manager). 0 disables PDF-based rotation.
PDFS_PER_SHARD = int(os.getenv("PDFS_PER_SHARD", "0"))


# Load .env if present (no override of shell env)
try:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)
except Exception as _e:
    LOG.debug(f".env load skipped: {_e}")

# Re-evaluate env-based toggles AFTER .env load (they were initially read before .env populated the process environment)
SHARD_MAX_VECTORS = int(os.getenv("SHARD_MAX_VECTORS", str(SHARD_MAX_VECTORS)))
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", str(EMBED_WORKERS)))
FORCE_NEW_SHARD = os.getenv("FORCE_NEW_SHARD", "1" if FORCE_NEW_SHARD else "0") == "1"
RESET_ALL = os.getenv("RESET", "1" if RESET_ALL else "0") == "1"
FAISS_GPU_MODE = os.getenv("FAISS_GPU_MODE", FAISS_GPU_MODE).lower()
MULTI_INDEX_BY_COURT = os.getenv("MULTI_INDEX_BY_COURT", "1" if MULTI_INDEX_BY_COURT else "0") == "1"
PDFS_PER_SHARD = int(os.getenv("PDFS_PER_SHARD", str(PDFS_PER_SHARD)))
LOG.info(f"Effective config -> MULTI_INDEX_BY_COURT={MULTI_INDEX_BY_COURT} PDFS_PER_SHARD={PDFS_PER_SHARD} SHARD_MAX_VECTORS={SHARD_MAX_VECTORS} FORCE_NEW_SHARD={FORCE_NEW_SHARD}")


# -------------
# DATA CLASSES
# -------------
@dataclass
class ChunkMeta:
    vector_id: int
    doc_id: str
    source_path: str
    chunk_index: int
    year: int | None
    court: str | None
    shard: int | None
    created_at: float


# -------------
# UTIL: Text extraction & chunking
# -------------
def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a text-based PDF using pdfplumber."""
    with pdfplumber.open(path) as pdf:
        pages = []
        for p in pdf.pages:
            txt = p.extract_text()
            if txt:
                pages.append(txt)
    return "\n\n".join(pages)


def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    base = Path(filename).stem
    year_match = re.search(r"(19|20)\d{2}", base)
    year = int(year_match.group(0)) if year_match else None
    court_match = re.search(r"\b(SC|HC|HighCourt|Supreme|District)\b", base, re.I)
    court = court_match.group(0).upper() if court_match else None
    return {"doc_id": base, "year": year, "court": court}


def chunk_text_legal_aware(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\nSECTION", "\n\n\n", ".\n", ". "],
    )
    return splitter.split_text(text)


# -------------
# METADATA DB (SQLite)
# -------------
def init_metadata_db(path: Path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            vector_id INTEGER PRIMARY KEY,
            doc_id TEXT,
            source_path TEXT,
            chunk_index INTEGER,
            year INTEGER,
            court TEXT,
            shard INTEGER,
            created_at REAL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_year ON chunks(year);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id);")
    # Backfill: add shard column if missing
    try:
        cur.execute("ALTER TABLE chunks ADD COLUMN shard INTEGER;")
    except Exception:
        pass
    conn.commit()
    conn.close()


def insert_chunk_meta(path: Path, metas: List[ChunkMeta]):
    if not metas:
        return
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    rows = [
        (
            m.vector_id,
            m.doc_id,
            m.source_path,
            m.chunk_index,
            m.year,
            m.court,
            m.shard,
            m.created_at,
        )
        for m in metas
    ]
    cur.executemany(
        "INSERT INTO chunks (vector_id, doc_id, source_path, chunk_index, year, court, shard, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# -------------
# FAISS SHARD HANDLER
# -------------
class FaissShardManager:
    """
    Manage multiple FAISS shards saved to disk. Each shard consists of:
      - index file: shard_{n}.index
      - metadata mapping persisted in sqlite (we keep vector_id stable across shards)
    """

    def __init__(self, out_dir: Path, embed_dim: int, idx_factory: str = INDEX_FACTORY, shard_max=SHARD_MAX_VECTORS):
        self.out_dir = out_dir
        self.embed_dim = embed_dim
        self.idx_factory = idx_factory
        self.shard_max = shard_max
        os.makedirs(out_dir, exist_ok=True)
        self.shards = self._discover_shards()
        self._next_vector_id = self._compute_next_vector_id()
        # PDF-based rotation counters
        self.pdfs_per_shard = PDFS_PER_SHARD
        self._current_shard_pdf_count = 0

    def _discover_shards(self) -> Dict[int, Dict]:
        shards: Dict[int, Dict] = {}
        for f in sorted(self.out_dir.glob("shard_*.index")):
            m = re.search(r"shard_(\d+)\.index", f.name)
            if not m:
                continue
            num = int(m.group(1))
            shards[num] = {"path": f, "index": None, "count": None}
        LOG.info(f"Discovered {len(shards)} shard files.")
        return shards

    def _compute_next_vector_id(self) -> int:
        if METADB_PATH.exists():
            conn = sqlite3.connect(METADB_PATH)
            cur = conn.cursor()
            try:
                cur.execute("SELECT MAX(vector_id) FROM chunks;")
                row = cur.fetchone()
            finally:
                conn.close()
            if row and row[0]:
                return int(row[0]) + 1
        return 1

    def _create_new_index(self) -> faiss.Index:
        # CPU default
        index = faiss.index_factory(self.embed_dim, self.idx_factory, faiss.METRIC_INNER_PRODUCT)
        if USE_ID_MAPPING:
            index = faiss.IndexIDMap(index)
        return index

    def _create_new_index_gpu_sharded(self) -> faiss.Index:
        # Build sharded container across all GPUs; distributes adds across devices
        ngpu = 0
        try:
            ngpu = faiss.get_num_gpus()
        except Exception:
            ngpu = 0
        if ngpu <= 0:
            raise RuntimeError("FAISS_GPU_MODE=gpu_sharded but no GPUs detected")
        shards = faiss.IndexShards(self.embed_dim, shallow_copy=False, own_fields=True)
        shards.shard = True
        base_cpu = faiss.index_factory(self.embed_dim, self.idx_factory, faiss.METRIC_INNER_PRODUCT)
        for dev in range(ngpu):
            gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, dev, base_cpu)
            shards.add_shard(gpu_index)
        if USE_ID_MAPPING:
            shards = faiss.IndexIDMap(shards)
        return shards

    def _create_new_index_gpu_replicated(self) -> faiss.Index:
        # Build replicated container across all GPUs; adds go to all devices
        ngpu = 0
        try:
            ngpu = faiss.get_num_gpus()
        except Exception:
            ngpu = 0
        if ngpu <= 0:
            raise RuntimeError("FAISS_GPU_MODE=gpu_replicated but no GPUs detected")
        replicas = faiss.IndexReplicas()
        base_cpu = faiss.index_factory(self.embed_dim, self.idx_factory, faiss.METRIC_INNER_PRODUCT)
        for dev in range(ngpu):
            gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, dev, base_cpu)
            replicas.addIndex(gpu_index)
        if USE_ID_MAPPING:
            replicas = faiss.IndexIDMap(replicas)
        return replicas

    def _load_index(self, shard_num: int):
        info = self.shards[shard_num]
        if info["index"] is None:
            LOG.info(f"Loading shard {shard_num} from disk: {info['path']}")
            idx = faiss.read_index(str(info["path"]))
            info["index"] = idx
            info["count"] = idx.ntotal
        return info["index"]

    def _create_shard_file(self, shard_num: int):
        p = self.out_dir / f"shard_{shard_num}.index"
        if FAISS_GPU_MODE == "gpu_sharded":
            idx = self._create_new_index_gpu_sharded()
        elif FAISS_GPU_MODE == "gpu_replicated":
            idx = self._create_new_index_gpu_replicated()
        else:
            idx = self._create_new_index()
        # Persist a CPU copy immediately (ensures file exists on disk)
        try:
            cpu_idx = faiss.index_gpu_to_cpu(idx) if hasattr(faiss, "index_gpu_to_cpu") and not isinstance(idx, faiss.IndexFlat) else idx
        except Exception:
            cpu_idx = self._create_new_index()  # fallback CPU index
        faiss.write_index(cpu_idx, str(p))
        self.shards[shard_num] = {"path": p, "index": idx, "count": 0}
        return idx

    def get_active_shard(self) -> Tuple[int, faiss.Index]:
        if not self.shards:
            self._create_shard_file(0)
        last = max(self.shards.keys())
        info = self.shards[last]
        if info["index"] is None:
            info["index"] = faiss.read_index(str(info["path"]))
            info["count"] = info["index"].ntotal
        if info["count"] is not None and info["count"] >= self.shard_max:
            new_num = last + 1
            self._create_shard_file(new_num)
            self._current_shard_pdf_count = 0
            return new_num, self.shards[new_num]["index"]
        return last, info["index"]

    def rotate_shard_on_pdf_boundary(self):
        """Rotate shard if PDF count threshold reached (called before adding vectors)."""
        if self.pdfs_per_shard and self._current_shard_pdf_count >= self.pdfs_per_shard:
            last = max(self.shards.keys()) if self.shards else -1
            new_num = last + 1
            LOG.info(f"PDFS_PER_SHARD reached ({self._current_shard_pdf_count} >= {self.pdfs_per_shard}) — creating shard {new_num}")
            self._current_shard_pdf_count = 0
            self._create_shard_file(new_num)

    def add_pdfs_count(self, n: int = 1):
        self._current_shard_pdf_count += int(n)

    def add_vectors(self, vectors: np.ndarray, ids: List[int], shard_num: int | None = None) -> int:
        if shard_num is None:
            shard_num, idx = self.get_active_shard()
        else:
            if shard_num not in self.shards:
                self._create_shard_file(shard_num)
            idx = self._load_index(shard_num)

        vecs = vectors.astype("float32")
        ids_np = np.array(ids, dtype="int64")
        idx.add_with_ids(vecs, ids_np)
        # Persist to disk: convert to CPU if GPU-backed
        try:
            cpu_idx = faiss.index_gpu_to_cpu(idx)
        except Exception:
            cpu_idx = idx
        faiss.write_index(cpu_idx, str(self.shards[shard_num]["path"]))
        self.shards[shard_num]["index"] = idx
        self.shards[shard_num]["count"] = idx.ntotal
        # Verify by reloading from disk to ensure persistence
        reloaded = faiss.read_index(str(self.shards[shard_num]["path"]))
        LOG.info(f"Added {len(ids)} vectors to shard {shard_num}; in-memory total {idx.ntotal}; reloaded total {reloaded.ntotal}")
        return shard_num

    def next_vector_id(self, count=1) -> List[int]:
        start = getattr(self, "_next_vector_id", 1)
        ids = list(range(start, start + count))
        self._next_vector_id = start + count
        return ids


# -------------
# EMBEDDINGS: Direct Google Generative AI client (batched loop)
# -------------
def _configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Add it to your environment or .env.")
    # Safe to call multiple times; configure is idempotent
    genai.configure(api_key=api_key)


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def get_gemini_embeddings_batch(texts: List[str]) -> np.ndarray:
    """Return array shape (N, EMBED_DIM), float32, L2-normalized using Gemini text-embedding-004.

    The google-generativeai Python client does not (currently) expose a pure batch endpoint; we loop.
    Resilient: on individual failures, substitutes a zero vector so overall batch size is preserved.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    _configure_genai()
    # Parallelize individual embed calls to maximize throughput
    vectors: List[List[float]] = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _one_embed(t: str) -> List[float]:
        try:
            resp = genai.embed_content(model="models/text-embedding-004", content=t)
            vec = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
            if vec is None:
                raise ValueError("No embedding field in response")
            return vec
        except Exception as e:
            # Avoid noisy logs in tight loops; keep minimal note
            LOG.debug(f"Embedding failed: {e}")
            return [0.0] * EMBED_DIM

    # Maintain order: collect futures with indices
    results: List[List[float]] = [None] * len(texts)  # type: ignore
    with ThreadPoolExecutor(max_workers=max(1, EMBED_WORKERS)) as ex:
        future_map = {ex.submit(_one_embed, t): i for i, t in enumerate(texts)}
        for fut in as_completed(future_map):
            i = future_map[fut]
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = [0.0] * EMBED_DIM
    vectors = results  # ordered
    arr = np.array(vectors, dtype=np.float32)
    # Pad or truncate if dimension mismatch (future-proofing)
    if arr.shape[1] < EMBED_DIM:
        pad_width = EMBED_DIM - arr.shape[1]
        arr = np.pad(arr, ((0,0),(0,pad_width)), mode="constant")
    elif arr.shape[1] > EMBED_DIM:
        arr = arr[:, :EMBED_DIM]
    arr = _l2_normalize(arr)
    return arr.astype("float32")


# -------------
# MAIN BATCHER: processes PDF files in batches, extracts, chunks, embeds, indexes
# -------------
def list_pdfs_to_process(pdf_dir: Path, processed_set: set) -> List[Path]:
    return [p for p in sorted(pdf_dir.glob("*.pdf")) if p.name not in processed_set]


def load_processed_log(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        return set(arr)
    except Exception:
        return set()


def save_processed_log(path: Path, processed: Iterable[str]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(processed), f)


def process_pdf_batch(pdf_paths: List[Path], shard_mgr: FaissShardManager, managers_by_court: Dict[str, FaissShardManager] | None = None) -> List[str]:
    """Processes a batch of PDF files, emits embeddings and metadata, and writes to FAISS shards."""
    all_text_chunks: List[str] = []
    provenance: List[Tuple[str, Dict[str, Any], int]] = []

    for pdfp in pdf_paths:
        try:
            text = extract_text_from_pdf(pdfp)
            if not text or text.strip() == "":
                LOG.warning(f"No text extracted: {pdfp}")
                continue
            if PRINT_PDF_TEXT:
                preview = text if len(text) <= PRINT_MAX_CHARS else text[:PRINT_MAX_CHARS] + "... [truncated]"
                LOG.info(f"\n===== PDF: {pdfp.name} | extracted {len(text)} chars =====\n{preview}\n===== END PDF: {pdfp.name} =====")
            meta = extract_metadata_from_filename(pdfp.name)
            chunks = chunk_text_legal_aware(text)
            if PRINT_CHUNK_PREVIEWS and chunks:
                LOG.info(f"Chunked into {len(chunks)} pieces; showing up to {CHUNK_PREVIEW_COUNT} previews:")
                for ci, ch in enumerate(chunks[:CHUNK_PREVIEW_COUNT]):
                    ch_prev = ch[:CHUNK_PREVIEW_CHARS] + ("..." if len(ch) > CHUNK_PREVIEW_CHARS else "")
                    LOG.info(f"  [chunk {ci}] {len(ch)} chars\n  {ch_prev}")
            # Count PDFs for rotation per manager (by court)
            if managers_by_court:
                court_key_pdf = (meta.get("court") or "OTHER").upper()
                mgr_pdf = managers_by_court.get(court_key_pdf, managers_by_court.get("OTHER", shard_mgr))
                mgr_pdf.add_pdfs_count(1)
                mgr_pdf.rotate_shard_on_pdf_boundary()

            for i, ch in enumerate(chunks):
                all_text_chunks.append(ch)
                provenance.append((pdfp.name, meta, i))
        except Exception as e:
            LOG.exception(f"Failed to parse {pdfp}: {e}")

    if not all_text_chunks:
        return []

    results_meta: List[ChunkMeta] = []
    failed_embeds = 0
    for i in range(0, len(all_text_chunks), BATCH_SIZE):
        batch_texts = all_text_chunks[i : i + BATCH_SIZE]
        LOG.info(f"Embedding batch {i // BATCH_SIZE + 1}: {len(batch_texts)} texts")
        vectors = get_gemini_embeddings_batch(batch_texts)
        # Count zero vectors (embedding failures) to surface issues
        failed_embeds += int(np.sum(np.isclose(vectors, 0.0))) // EMBED_DIM
        if failed_embeds:
            LOG.warning(f"Cumulative embedding fallbacks (zero vectors): {failed_embeds}")

        # Choose manager (multi-index by court if enabled)
        active_mgr = shard_mgr
        if managers_by_court:
            # Use the first provenance entry of this mini-batch to decide routing
            first_pdf_name, first_meta, _ = provenance[i]
            court_key = (first_meta.get("court") or "OTHER").upper()
            active_mgr = managers_by_court.get(court_key, managers_by_court.get("OTHER", shard_mgr))
        # Also check PDF-based rotation before adding vectors
        active_mgr.rotate_shard_on_pdf_boundary()

        ids = active_mgr.next_vector_id(len(batch_texts))
        # Optionally force a new shard per mini-batch
        if FORCE_NEW_SHARD:
            next_num = max(shard_mgr.shards.keys()) + 1 if shard_mgr.shards else 0
            LOG.info(f"FORCE_NEW_SHARD is ON — creating shard {next_num} for this batch")
            shard_mgr._create_shard_file(next_num)
            shard_num = active_mgr.add_vectors(vectors, ids, shard_num=next_num)
        else:
            shard_num = active_mgr.add_vectors(vectors, ids)

        meta_objs: List[ChunkMeta] = []
        ts = time.time()
        for j, vid in enumerate(ids):
            pdf_name, meta, chunk_index = provenance[i + j]
            cm = ChunkMeta(
                vector_id=vid,
                doc_id=meta["doc_id"],
                source_path=pdf_name,
                chunk_index=chunk_index,
                year=meta.get("year"),
                court=meta.get("court"),
                shard=shard_num,
                created_at=ts,
            )
            meta_objs.append(cm)
        insert_chunk_meta(METADB_PATH, meta_objs)
        results_meta.extend(meta_objs)

    return [p.name for p in pdf_paths]


# -------------
# RETRIEVAL: query across shards and merge (+ optional rerank)
# -------------
def _load_chunk_text_from_pdf(source_path: str, chunk_index: int) -> str | None:
    pdf_path = PDF_DIR / source_path
    if not pdf_path.exists():
        return None
    try:
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text_legal_aware(text)
        if 0 <= chunk_index < len(chunks):
            return chunks[chunk_index]
    except Exception:
        return None
    return None


def _candidate_ids(filters: Dict[str, Any]) -> List[int]:
    """Return vector_ids matching metadata filters.

    Supported keys: year_min, year_max, court, doc_id_like, source_like, shard_in (list[int]).
    """
    if not filters:
        return []
    conn = sqlite3.connect(METADB_PATH)
    cur = conn.cursor()
    clauses = []
    params: List[Any] = []
    if "year_min" in filters:
        clauses.append("year >= ?")
        params.append(int(filters["year_min"]))
    if "year_max" in filters:
        clauses.append("year <= ?")
        params.append(int(filters["year_max"]))
    if "court" in filters and filters["court"]:
        clauses.append("UPPER(court) = UPPER(?)")
        params.append(str(filters["court"]))
    if "doc_id_like" in filters and filters["doc_id_like"]:
        clauses.append("doc_id LIKE ?")
        params.append(str(filters["doc_id_like"]))
    if "source_like" in filters and filters["source_like"]:
        clauses.append("source_path LIKE ?")
        params.append(str(filters["source_like"]))
    if "shard_in" in filters and filters["shard_in"]:
        placeholders = ",".join(["?"] * len(filters["shard_in"]))
        clauses.append(f"shard IN ({placeholders})")
        params.extend([int(x) for x in filters["shard_in"]])
    where = " AND ".join(clauses) if clauses else ""
    q = "SELECT vector_id FROM chunks" + (f" WHERE {where}" if where else "")
    cur.execute(q, params)
    ids = [int(row[0]) for row in cur.fetchall()]
    conn.close()
    return ids


def retrieve(query: str, shard_mgr: FaissShardManager, top_k: int = 5, search_shards: List[int] | None = None, rerank_with_gemini: bool = False, filters: Dict[str, Any] | None = None):
    # 1) embed query
    q_vec = get_gemini_embeddings_batch([query])[0].astype("float32").reshape(1, -1)
    # Prefilter by metadata if provided
    allowed_ids = set(_candidate_ids(filters or {})) if filters else set()
    results: List[Dict[str, Any]] = []
    shard_keys = sorted(shard_mgr.shards.keys()) if search_shards is None else search_shards
    for s in shard_keys:
        try:
            idx = shard_mgr._load_index(s)
        except Exception as e:
            LOG.warning(f"Failed to load shard {s}: {e}")
            continue
        D, I = idx.search(q_vec, top_k)
        for dist, vid in zip(D[0], I[0]):
            if vid == -1:
                continue
            if allowed_ids and int(vid) not in allowed_ids:
                continue
            results.append({"shard": s, "vector_id": int(vid), "score": float(dist)})
    if not results:
        return []

    results = sorted(results, key=lambda r: r["score"], reverse=True)[: top_k]

    # fetch metadata
    conn = sqlite3.connect(METADB_PATH)
    cur = conn.cursor()
    enriched: List[Dict[str, Any]] = []
    for r in results:
        cur.execute(
            "SELECT doc_id, source_path, chunk_index, year, court, shard FROM chunks WHERE vector_id = ?",
            (r["vector_id"],),
        )
        row = cur.fetchone()
        if not row:
            continue
        doc_id, source_path, chunk_index, year, court, shard = row
        enriched.append(
            {
                "vector_id": r["vector_id"],
                "score": r["score"],
                "doc_id": doc_id,
                "source_path": source_path,
                "chunk_index": int(chunk_index),
                "year": year,
                "court": court,
                "shard": shard,
            }
        )
    conn.close()

    if not rerank_with_gemini:
        return enriched

    # Optional rerank with Gemini by re-embedding the actual chunk texts
    chunk_texts: List[str] = []
    for item in enriched:
        txt = _load_chunk_text_from_pdf(item["source_path"], item["chunk_index"]) or ""
        chunk_texts.append(txt)

    if any(t for t in chunk_texts):
        qv = get_gemini_embeddings_batch([query])[0]
        dv = get_gemini_embeddings_batch(chunk_texts)
        # cosine via inner product on normalized vectors
        sims = (dv @ qv)
        for i, sim in enumerate(sims.tolist()):
            enriched[i]["rerank_score"] = float(sim)
        enriched.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    return enriched


# -------------
# main orchestration
# -------------
def run_batch_pipeline():
    if not PDF_DIR.exists():
        raise FileNotFoundError(f"PDF_DIR not found: {PDF_DIR}")
    # Optional full reset
    if RESET_ALL:
        LOG.warning("RESET=1 detected: clearing shards and metadata for a full re-ingestion …")
        try:
            # remove shard files
            if OUTPUT_DIR.exists():
                for f in OUTPUT_DIR.glob("shard_*.index"):
                    f.unlink(missing_ok=True)
            else:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            # remove metadata DB and processed log
            if METADB_PATH.exists():
                METADB_PATH.unlink(missing_ok=True)
            if PROCESSED_LOG.exists():
                PROCESSED_LOG.unlink(missing_ok=True)
        except Exception as e:
            LOG.exception(f"Reset failed: {e}")
    init_metadata_db(METADB_PATH)
    processed = load_processed_log(PROCESSED_LOG)
    # Build managers
    if MULTI_INDEX_BY_COURT:
        out_sc = Path(__file__).with_name("faiss_shards_sc")
        out_hc = Path(__file__).with_name("faiss_shards_hc")
        out_other = Path(__file__).with_name("faiss_shards_other")
        managers_by_court = {
            "SC": FaissShardManager(out_sc, EMBED_DIM),
            "HC": FaissShardManager(out_hc, EMBED_DIM),
            "OTHER": FaissShardManager(out_other, EMBED_DIM),
        }
        shard_mgr = managers_by_court["OTHER"]
    else:
        managers_by_court = None
        shard_mgr = FaissShardManager(OUTPUT_DIR, EMBED_DIM)
    # If there are no shard files but processed log is non-empty, vectors were lost.
    # In that case, reprocess everything by resetting the processed set in-memory.
    if not shard_mgr.shards and processed and not MULTI_INDEX_BY_COURT:
        LOG.warning("No shard files found but processed.json lists entries — reprocessing all PDFs.")
        processed = set()
    all_pdfs = list_pdfs_to_process(PDF_DIR, processed)
    LOG.info(f"{len(all_pdfs)} PDFs to consider, {len(processed)} already processed.")

    for i in range(0, len(all_pdfs), PDF_BATCH):
        batch_files = all_pdfs[i : i + PDF_BATCH]
        try:
            processed_names = process_pdf_batch(batch_files, shard_mgr, managers_by_court=managers_by_court)
            processed.update(processed_names)
            save_processed_log(PROCESSED_LOG, processed)
            LOG.info(f"Processed batch {i // PDF_BATCH + 1}. Total processed files: {len(processed)}")
        except Exception as e:
            LOG.exception(f"Batch failed: {e}")
    LOG.info("Batch pipeline finished.")


if __name__ == "__main__":
    # Run ingestion pipeline (requires GEMINI_API_KEY)
    try:
        # Debug FAISS GPU environment
        try:
            ngpu = faiss.get_num_gpus()
            LOG.info(f"FAISS GPUs detected: {ngpu}; mode={FAISS_GPU_MODE}")
        except Exception as e:
            LOG.info(f"FAISS GPU query failed: {e}; mode={FAISS_GPU_MODE}")
        run_batch_pipeline()
        LOG.info("Example retrieval (top 5):")
        mgr = FaissShardManager(OUTPUT_DIR, EMBED_DIM)
        # Print index container types and shard listing
        for s, info in sorted(mgr.shards.items()):
            idx = mgr._load_index(s)
            LOG.info(f"Shard {s}: type={type(idx).__name__} ntotal={getattr(idx,'ntotal',None)} path={info['path']}")
            if isinstance(idx, faiss.IndexShards):
                LOG.info(f"  IndexShards: shard_flag={getattr(idx,'shard',None)}")
            if isinstance(idx, faiss.IndexReplicas):
                LOG.info("  IndexReplicas active")
        res = retrieve("fundamental rights and article 21", mgr, top_k=5, rerank_with_gemini=True)
        for r in res:
            LOG.info(f"vid={r['vector_id']} score={r.get('rerank_score', r['score']):.4f} source={r['source_path']} chunk={r['chunk_index']}")
            # Print a short text preview for the retrieved chunk
            _txt = _load_chunk_text_from_pdf(r["source_path"], r["chunk_index"]) or ""
            if _txt:
                prev = _txt[:CHUNK_PREVIEW_CHARS] + ("..." if len(_txt) > CHUNK_PREVIEW_CHARS else "")
                LOG.info(f"  TEXT PREVIEW: {prev}")
    except RuntimeError as e:
        LOG.error(str(e))
        LOG.error("Set GEMINI_API_KEY in your environment or .env, then rerun.")
    except Exception:
        raise

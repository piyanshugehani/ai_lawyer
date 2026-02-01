import os
import time
import logging
import re
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any

# Third-party imports
try:
    import pdfplumber
    from pinecone import Pinecone
    from google import genai
    from google.genai import types
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tqdm import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    raise ImportError(f"Missing library: {e}. Run: pip install pdfplumber pinecone-client google-genai langchain-text-splitters tqdm python-dotenv")

# ------------------------------
# 1. Configuration & Setup
# ------------------------------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Env Variables / Config
INDEX_NAME = "legal-judgments-index"
MULTI_INDEX = os.getenv("MULTI_INDEX_BY_COURT", "0") == "1"
RESET_MODE = os.getenv("RESET", "0") == "1"
CPU_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))

DATA_DIR = Path(__file__).parent / "supreme-court-pdfs-05-06"
PROCESSED_TRACKER_FILE = Path(__file__).parent / "processed.json"

# Tuning
PINECONE_BATCH_SIZE = 100
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
GEMINI_RPM_DELAY = 4.0

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("execution.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing API Keys. Check your .env file.")

# ------------------------------
# 2. Initialize Clients
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = genai.Client(api_key=GEMINI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ------------------------------
# 3. Helper Functions
# ------------------------------
def get_processed_files() -> set:
    if PROCESSED_TRACKER_FILE.exists() and not RESET_MODE:
        try:
            with open(PROCESSED_TRACKER_FILE, "r") as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def mark_files_as_processed(filenames: List[str]):
    current = get_processed_files()
    current.update(filenames)
    with open(PROCESSED_TRACKER_FILE, "w") as f:
        json.dump(list(current), f)

def sanitize_id(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', text)

def get_namespace(filename: str) -> str:
    if not MULTI_INDEX: return ""
    if filename.lower().startswith("s_") or "scc" in filename.lower():
        return "supreme_court"
    return "high_court"

# ------------------------------
# 4. Core Logic
# ------------------------------

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parses PDF, Cleans text (Lower/Trim), and returns Chunks.
    """
    file_name = pdf_path.name
    results = []
    
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    full_text += txt + " " # Append with space to avoid merging words across lines

        # --- OPTIMIZATION 2: Print Initial Raw Text ---
        if full_text:
            # Print a snippet of the raw text to console for verification
            preview = full_text[:300].replace('\n', ' ')
            print(f"\n📄 [Preview {file_name}]: {preview}...\n")
        
        # --- OPTIMIZATION 1: Cleaning Logic ---
        # 1. Normalize whitespace (remove tabs, newlines, double spaces)
        # 2. Lowercase everything
        cleaned_text = re.sub(r'\s+', ' ', full_text).strip().lower()
        
        if not cleaned_text:
            return []

        # Split the CLEANED text
        chunks = splitter.split_text(cleaned_text)
        namespace = get_namespace(file_name)
        
        match = re.search(r'(19|20)\d{2}', file_name)
        year = int(match.group(0)) if match else 0

        for i, chunk in enumerate(chunks):
            unique_id = f"{sanitize_id(file_name)}_c{i}"
            results.append({
                "id": unique_id,
                "text": chunk,
                "namespace": namespace,
                "metadata": {
                    "filename": file_name,
                    "year": year,
                    "chunk_index": i,
                    "text": chunk 
                }
            })
    except Exception as e:
        logger.warning(f"⚠️ Corrupt/Skipped: {file_name} ({e})")
        return []
        
    return results

def batch_embed_and_upload(chunk_buffer: List[Dict[str, Any]]):
    """
    Groups chunks, Embeds (Gemini), and Upserts (Pinecone).
    """
    if not chunk_buffer:
        return

    from collections import defaultdict
    chunks_by_ns = defaultdict(list)
    for c in chunk_buffer:
        chunks_by_ns[c['namespace']].append(c)

    for ns, ns_chunks in chunks_by_ns.items():
        for i in range(0, len(ns_chunks), PINECONE_BATCH_SIZE):
            batch = ns_chunks[i : i + PINECONE_BATCH_SIZE]
            texts = [x['text'] for x in batch]
            
            if not texts: continue

            try:
                # Embed (Gemini)
                response = client.models.embed_content(
                    model="models/text-embedding-004", 
                    contents=texts,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                
                # Match IDs
                vectors = []
                if hasattr(response, 'embeddings'):
                    for idx, embedding_obj in enumerate(response.embeddings):
                        item = batch[idx]
                        vectors.append({
                            "id": item['id'],
                            "values": embedding_obj.values,
                            "metadata": item['metadata']
                        })
                
                # Upsert (Pinecone)
                if vectors:
                    index.upsert(vectors=vectors, namespace=ns)
                
                time.sleep(GEMINI_RPM_DELAY)
                
            except Exception as e:
                if "429" in str(e):
                    logger.error("🛑 Rate Limit Hit (429). Sleeping 60s...")
                    time.sleep(60)
                else:
                    logger.error(f"❌ Batch Error: {e}")

# ------------------------------
# 5. Main Loop
# ------------------------------
def main():
    if RESET_MODE and PROCESSED_TRACKER_FILE.exists():
        os.remove(PROCESSED_TRACKER_FILE)
        logger.info("♻️  Reset mode: Tracker cleared.")

    if not DATA_DIR.exists():
        logger.error(f"❌ Data directory not found: {DATA_DIR}")
        return

    all_pdfs = list(DATA_DIR.rglob("*.pdf"))
    processed = get_processed_files()
    new_pdfs = [p for p in all_pdfs if p.name not in processed]
    
    total_new = len(new_pdfs)
    logger.info(f"🚀 Processing {total_new} new PDFs (Total Found: {len(all_pdfs)})")

    FILE_BATCH_SIZE = 20 
    
    with tqdm(total=total_new, desc="Progress") as pbar:
        for i in range(0, total_new, FILE_BATCH_SIZE):
            batch_files = new_pdfs[i : i + FILE_BATCH_SIZE]
            current_filenames = [p.name for p in batch_files]
            
            chunk_buffer = []

            # Step A: Parallel Parsing + Cleaning
            with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
                results = list(executor.map(extract_text_from_pdf, batch_files))
                for res in results:
                    if res: chunk_buffer.extend(res)

            # Step B: Sequential Upload
            if chunk_buffer:
                batch_embed_and_upload(chunk_buffer)
            
            # Step C: Checkpoint
            mark_files_as_processed(current_filenames)
            pbar.update(len(batch_files))

    logger.info("✅ All PDF processing complete.")

if __name__ == "__main__":
    main()
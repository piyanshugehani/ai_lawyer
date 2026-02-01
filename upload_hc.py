import os
import time
import logging
import re
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple

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
INDEX_NAME = "legal-judgments-index" # Using the same index, but different Namespace
NAMESPACE = "high_court"             # Separate Namespace for HC
RESET_MODE = os.getenv("RESET", "0") == "1"
CPU_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))

DATA_DIR = Path(__file__).parent / "high-court-pdfs-05-06"
PROCESSED_TRACKER_FILE = Path(__file__).parent / "processed_hc.json"

# Tuning
PINECONE_BATCH_SIZE = 100
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
GEMINI_RPM_DELAY = 4.0 # 4s delay to stay safe

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("execution_hc.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing API Keys. Check your .env file.")

# ------------------------------
# 2. Domain Logic: High Court Mappings
# ------------------------------
HC_STATE_MAP = {
    "Allahabad": "Uttar Pradesh", "Andhra Pradesh": "Andhra Pradesh", "Bombay": "Maharashtra",
    "Calcutta": "West Bengal", "Chhattisgarh": "Chhattisgarh", "Delhi": "Delhi",
    "Gauhati": "Assam", "Gujarat": "Gujarat", "Himachal Pradesh": "Himachal Pradesh",
    "Jammu": "Jammu & Kashmir", "Kashmir": "Jammu & Kashmir", "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka", "Kerala": "Kerala", "Madhya Pradesh": "Madhya Pradesh",
    "Madras": "Tamil Nadu", "Manipur": "Manipur", "Meghalaya": "Meghalaya",
    "Orissa": "Odisha", "Odisha": "Odisha", "Patna": "Bihar",
    "Punjab": "Punjab & Haryana", "Haryana": "Punjab & Haryana", "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim", "Telangana": "Telangana", "Tripura": "Tripura",
    "Uttarakhand": "Uttarakhand"
}

# ------------------------------
# 3. Initialize Clients
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = genai.Client(api_key=GEMINI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ------------------------------
# 4. Helper Functions
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

def detect_court_metadata(text_sample: str, filename: str) -> Tuple[str, str]:
    """Returns (Court Name, State) based on text headers or filename."""
    text_lower = text_sample.lower()[:2000] if text_sample else ""
    filename_lower = filename.lower()
    
    for key, state in HC_STATE_MAP.items():
        key_lower = key.lower()
        # Check Text or Filename
        if f"high court of {key_lower}" in text_lower or f"{key_lower} high court" in text_lower or key_lower in filename_lower:
            return f"{key} High Court", state

    return "Unknown High Court", "Unknown"

# ------------------------------
# 5. Core Logic
# ------------------------------

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Parses PDF -> Fixes OCR Artifacts -> Detects Metadata -> Cleans -> Chunks.
    """
    file_name = pdf_path.name
    results = []
    
    try:
        full_text = ""
        # We need specific page 1 text for metadata detection
        page_one_text = "" 

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                txt = page.extract_text()
                if txt:
                    if i == 0: page_one_text = txt
                    full_text += txt + " "

        if not full_text.strip():
            return []

        # --- STEP 1: Fix specific OCR artifacts (like "IIINNN" -> "IN") ---
        # Matches uppercase letter repeated 3+ times
        fixed_text = re.sub(r'([A-Z])\1\1+', r'\1', full_text)
        
        # --- STEP 2: Detect Metadata (using text with case preserved ideally, or lightly cleaned) ---
        detected_court, detected_state = detect_court_metadata(fixed_text, file_name)

        # --- STEP 3: Final Cleaning (Lowercasing + Whitespace) ---
        cleaned_text = re.sub(r'\s+', ' ', fixed_text).strip().lower()

        # Preview for debugging
        if len(cleaned_text) > 0:
            print(f"\n📄 [{detected_court}]: {cleaned_text[:200]}...\n")

        # Split
        chunks = splitter.split_text(cleaned_text)
        
        match = re.search(r'(19|20)\d{2}', file_name)
        year = int(match.group(0)) if match else 0

        for i, chunk in enumerate(chunks):
            unique_id = f"hc_{sanitize_id(file_name)}_c{i}"
            results.append({
                "id": unique_id,
                "text": chunk,
                "namespace": NAMESPACE,
                "metadata": {
                    "filename": file_name,
                    "year": year,
                    "court": detected_court,
                    "state": detected_state,
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
    Optimized: Batches texts -> One API Call -> Upsert.
    """
    if not chunk_buffer:
        return

    # Process in batches of 100
    for i in range(0, len(chunk_buffer), PINECONE_BATCH_SIZE):
        batch = chunk_buffer[i : i + PINECONE_BATCH_SIZE]
        texts = [x['text'] for x in batch]
        
        if not texts: continue

        try:
            # 1. Embed (Gemini)
            response = client.models.embed_content(
                model="models/text-embedding-004", 
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            # 2. Map Responses
            vectors = []
            if hasattr(response, 'embeddings'):
                for idx, embedding_obj in enumerate(response.embeddings):
                    item = batch[idx]
                    vectors.append({
                        "id": item['id'],
                        "values": embedding_obj.values,
                        "metadata": item['metadata']
                    })
            
            # 3. Upsert (Pinecone)
            if vectors:
                index.upsert(vectors=vectors, namespace=NAMESPACE)
            
            # 4. Safety Sleep
            time.sleep(GEMINI_RPM_DELAY)
            
        except Exception as e:
            if "429" in str(e):
                logger.error("🛑 Rate Limit Hit (429). Sleeping 60s...")
                time.sleep(60)
            else:
                logger.error(f"❌ Batch Error: {e}")

# ------------------------------
# 6. Main Orchestration
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
    logger.info(f"🚀 Processing {total_new} High Court PDFs (Total Found: {len(all_pdfs)})")

    FILE_BATCH_SIZE = 20 
    
    with tqdm(total=total_new, desc="Processing HC PDFs") as pbar:
        for i in range(0, total_new, FILE_BATCH_SIZE):
            batch_files = new_pdfs[i : i + FILE_BATCH_SIZE]
            current_filenames = [p.name for p in batch_files]
            
            chunk_buffer = []

            # Step A: Parallel Parsing + Heuristics + Cleaning
            with ProcessPoolExecutor(max_workers=CPU_WORKERS) as executor:
                results = list(executor.map(extract_text_from_pdf, batch_files))
                for res in results:
                    if res: chunk_buffer.extend(res)

            # Step B: Sequential Optimized Upload
            if chunk_buffer:
                batch_embed_and_upload(chunk_buffer)
            
            # Step C: Checkpoint
            mark_files_as_processed(current_filenames)
            pbar.update(len(batch_files))

    logger.info("✅ High Court processing complete.")

if __name__ == "__main__":
    main()
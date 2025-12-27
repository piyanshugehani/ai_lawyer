import os
import time
import logging
import re
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

import pdfplumber
from pinecone import Pinecone
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ------------------------------
# Environment & Config
# ------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
except ImportError:
    pass

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    raise RuntimeError("❌ Missing API keys in .env")

# ------------------------------
# Pinecone / Dataset Settings
# ------------------------------
INDEX_NAME = "high-court-judgments-index"
DATA_DIR = Path(__file__).with_name("high-court-pdfs-05-06")

PROCESSED_TRACKER_FILE = "processed_hc.json"

# ------------------------------
# Performance Tuning
# ------------------------------
MAX_WORKERS_PDF = 4
MAX_WORKERS_EMBED = 8
PINECONE_BATCH_SIZE = 100

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ------------------------------
# Logging (HC-specific)
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("execution_errors_hc.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Clients
# ------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = genai.Client(api_key=GEMINI_API_KEY)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ------------------------------
# Processed Files Tracker
# ------------------------------
def get_processed_files() -> set:
    if os.path.exists(PROCESSED_TRACKER_FILE):
        try:
            with open(PROCESSED_TRACKER_FILE, "r") as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Failed reading processed_hc.json: {e}")
    return set()

def mark_files_as_processed(filenames: List[str]):
    processed = get_processed_files()
    processed.update(filenames)
    try:
        with open(PROCESSED_TRACKER_FILE, "w") as f:
            json.dump(list(processed), f, indent=4)
    except Exception as e:
        logger.error(f"Failed updating processed_hc.json: {e}")

# ------------------------------
# Text Cleaning & Heuristics
# ------------------------------
HC_STATE_MAP = {
    "Allahabad": "Uttar Pradesh",
    "Andhra Pradesh": "Andhra Pradesh",
    "Bombay": "Maharashtra",
    "Calcutta": "West Bengal",
    "Chhattisgarh": "Chhattisgarh",
    "Delhi": "Delhi",
    "Gauhati": "Assam",
    "Gujarat": "Gujarat",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu": "Jammu & Kashmir",
    "Kashmir": "Jammu & Kashmir",
    "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Madhya Pradesh": "Madhya Pradesh",
    "Madras": "Tamil Nadu",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Orissa": "Odisha",
    "Odisha": "Odisha",
    "Patna": "Bihar",
    "Punjab": "Punjab & Haryana",
    "Haryana": "Punjab & Haryana",
    "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim",
    "Telangana": "Telangana",
    "Tripura": "Tripura",
    "Uttarakhand": "Uttarakhand"
}

def clean_text(text: str) -> str:
    """
    Cleans common OCR artifacts found in Indian High Court PDFs.
    """
    if not text:
        return ""

    # 1. Fix "Simulated Bold" artifacts (e.g., "IIINNN" -> "IN", "TTTHHHEEE" -> "THE")
    # Matches any uppercase letter repeated 3 times or more
    text = re.sub(r'([A-Z])\1\1+', r'\1', text)

    # 2. Normalize whitespace (tabs/newlines -> single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_court_metadata(text_sample: str, filename: str) -> Tuple[str, str]:
    """
    Returns (Court Name, State) based on text headers or filename using simple heuristics.
    """
    text_lower = text_sample.lower()[:2000] if text_sample else "" # Only scan header area
    filename_lower = filename.lower()
    
    # Priority: Check explicit "High Court of [Place]" in text
    for key, state in HC_STATE_MAP.items():
        key_lower = key.lower()
        
        # Check 1: Text pattern
        if f"high court of {key_lower}" in text_lower or f"{key_lower} high court" in text_lower:
            return f"{key} High Court", state
        
        # Check 2: Filename pattern
        if key_lower in filename_lower:
            return f"{key} High Court", state

    return "Unknown High Court", "Unknown"

# ------------------------------
# PDF → Chunk Extraction
# ------------------------------
def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    results = []
    file_name = pdf_path.name

    # Metadata placeholders
    detected_court = "Unknown High Court"
    detected_state = "Unknown"

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # 1. First Pass: Detect Metadata from Page 1 (Header usually here)
            if len(pdf.pages) > 0:
                # IMPORTANT: Extract AND Clean before detection
                # If we don't clean, "IIINNN TTTHHHEEE HIGH COURT" won't match our heuristics
                raw_p1 = pdf.pages[0].extract_text()
                cleaned_p1 = clean_text(raw_p1)
                detected_court, detected_state = detect_court_metadata(cleaned_p1, file_name)

            # 2. Second Pass: Extract and Chunk all pages
            for page_no, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text()
                
                # Apply Cleaning
                text = clean_text(raw_text)

                if not text:
                    continue

                chunks = splitter.split_text(text)

                match = re.search(r"(19|20)\d{2}", file_name)
                year = int(match.group()) if match else 0

                for idx, chunk in enumerate(chunks):
                    results.append({
                        "id": f"hc_{file_name}_p{page_no}_c{idx}",
                        "text": chunk,
                        "metadata": {
                            "court_level": "High Court",
                            "court": detected_court,   # Heuristic applied
                            "state": detected_state,   # Heuristic applied
                            "year": year,
                            "filename": file_name,
                            "page": page_no,
                            "chunk_index": idx,
                            "source": "pdf",
                            "text": chunk
                        }
                    })

    except Exception as e:
        logger.error(f"❌ Failed parsing PDF {file_name}: {e}")

    return results

# ------------------------------
# Embedding Generation
# ------------------------------
def generate_embedding(text: str, retry_count: int = 0):
    try:
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        if resp.embeddings:
            return resp.embeddings[0].values

    except Exception as e:
        # Simple backoff for rate limits
        if "429" in str(e) and retry_count < 5:
            time.sleep((2 ** retry_count) + 1)
            return generate_embedding(text, retry_count + 1)
        logger.error(f"Embedding failed: {e}")

    return None

# ------------------------------
# Embed + Upsert
# ------------------------------
def process_batch_embeddings_and_upsert(chunks: List[Dict[str, Any]]):
    vectors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBED) as executor:
        future_map = {
            executor.submit(generate_embedding, c["text"]): c
            for c in chunks
        }

        for future in as_completed(future_map):
            chunk = future_map[future]
            embedding = future.result()
            if embedding:
                vectors.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": chunk["metadata"]
                })

    if not vectors:
        return

    # Batch upsert to Pinecone
    for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
        try:
            index.upsert(vectors=vectors[i:i + PINECONE_BATCH_SIZE])
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")

# ------------------------------
# Main Orchestration
# ------------------------------
def main():
    if not DATA_DIR.exists():
        logger.error(f"Directory not found: {DATA_DIR}")
        return

    all_pdfs = list(DATA_DIR.rglob("*.pdf"))
    processed = get_processed_files()
    pdfs_to_process = [p for p in all_pdfs if p.name not in processed]

    logger.info(f"📄 Total PDFs found: {len(all_pdfs)}")
    logger.info(f"⏭️ Skipped (already processed): {len(processed)}")
    logger.info(f"🚀 Processing new PDFs: {len(pdfs_to_process)}")

    if not pdfs_to_process:
        logger.info("Nothing to process.")
        return

    PDF_BATCH = 50

    with tqdm(total=len(pdfs_to_process), desc="High Court PDFs") as pbar:
        for i in range(0, len(pdfs_to_process), PDF_BATCH):
            batch = pdfs_to_process[i:i + PDF_BATCH]
            filenames = [p.name for p in batch]

            extracted_chunks = []

            # 1. Parallel PDF Extraction (with Cleaning & Heuristics)
            with ProcessPoolExecutor(max_workers=MAX_WORKERS_PDF) as executor:
                for result in executor.map(extract_text_from_pdf, batch):
                    extracted_chunks.extend(result)
                    pbar.update(1)

            # 2. Parallel Embedding & Upsert
            for j in range(0, len(extracted_chunks), PINECONE_BATCH_SIZE):
                process_batch_embeddings_and_upsert(
                    extracted_chunks[j:j + PINECONE_BATCH_SIZE]
                )

            # 3. Mark batch as processed
            mark_files_as_processed(filenames)

    logger.info("✅ High Court ingestion complete.")

# ------------------------------
# Entry
# ------------------------------
if __name__ == "__main__":
    main()
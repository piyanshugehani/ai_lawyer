import os
import time
import logging
import re
import json  # <--- Added json import
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pdfplumber
from pinecone import Pinecone
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# ------------------------------
# Configuration & Setup
# ------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
except ImportError:
    pass

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-judgments-index"
DATA_DIR = Path(__file__).with_name("supreme-court-pdfs-05-06")
PROCESSED_TRACKER_FILE = "processed.json"  # <--- Tracker File

# Tuning Parameters
MAX_WORKERS_PDF = 4
MAX_WORKERS_EMBED = 8
PINECONE_BATCH_SIZE = 100
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# Logging (Keeps a log for errors/info, but NOT for tracking state)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("execution_errors.log"), # Renamed to avoid confusion
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing API Keys in .env")

# Initialize Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = genai.Client(api_key=GEMINI_API_KEY)

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# ------------------------------
# JSON State Management Functions
# ------------------------------
def get_processed_files() -> set:
    """Reads processed.json and returns a set of filenames."""
    if os.path.exists(PROCESSED_TRACKER_FILE):
        try:
            with open(PROCESSED_TRACKER_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return set(data)
        except Exception as e:
            logger.error(f"Error reading processed.json: {e}")
    return set()

def mark_files_as_processed(new_filenames: List[str]):
    """Updates processed.json with new filenames."""
    current_set = get_processed_files()
    current_set.update(new_filenames)
    try:
        with open(PROCESSED_TRACKER_FILE, "w") as f:
            json.dump(list(current_set), f, indent=4)
    except Exception as e:
        logger.error(f"Failed to update processed.json: {e}")

# ------------------------------
# Core Functions
# ------------------------------

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """CPU-bound task: Extracts text, finds year via Regex, splits into chunks."""
    results = []
    file_name = pdf_path.name
    
    try:
        text_by_page = {}
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    txt = page.extract_text()
                    if txt and txt.strip():
                        text_by_page[i + 1] = txt
                except Exception:
                    pass

        for page_num, page_text in text_by_page.items():
            chunks = splitter.split_text(page_text)
            for i, chunk in enumerate(chunks):
                
                # Metadata Logic
                if file_name.lower().startswith("s_") or "scc" in file_name.lower():
                    cat = "scc"
                else:
                    cat = "hcc"
                
                match = re.search(r'(19|20)\d{2}', file_name)
                year = int(match.group(0)) if match else 0

                results.append({
                    "id": f"{file_name}_p{page_num}_c{i}",
                    "text": chunk,
                    "metadata": {
                        "category": cat,
                        "year": year,
                        "filename": file_name,
                        "page": page_num,
                        "chunk_index": i,
                        "text": chunk 
                    }
                })
    except Exception as e:
        logger.error(f"Failed to parse PDF {file_name}: {e}")
        return []
    
    return results

def generate_embedding(text: str, retry_count=0):
    """IO-bound task: Calls Gemini API with exponential backoff."""
    try:
        resp = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        if hasattr(resp, 'embeddings') and len(resp.embeddings) > 0:
            return resp.embeddings[0].values
            
    except Exception as e:
        if "429" in str(e) and retry_count < 5:
            wait_time = (2 ** retry_count) + 1
            time.sleep(wait_time)
            return generate_embedding(text, retry_count + 1)
        
        logger.error(f"Embedding failed (attempt {retry_count}): {e}")
    return None

def process_batch_embeddings_and_upsert(chunks: List[Dict[str, Any]]):
    """Embeds via ThreadPool and upserts to Pinecone."""
    if not chunks:
        return

    vectors_to_upsert = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBED) as executor:
        future_to_chunk = {executor.submit(generate_embedding, c['text']): c for c in chunks}
        
        for future in as_completed(future_to_chunk):
            chunk_data = future_to_chunk[future]
            embedding = future.result()
            
            if embedding:
                vectors_to_upsert.append({
                    "id": chunk_data['id'],
                    "values": embedding,
                    "metadata": chunk_data['metadata']
                })

    if vectors_to_upsert:
        for i in range(0, len(vectors_to_upsert), PINECONE_BATCH_SIZE):
            batch = vectors_to_upsert[i : i + PINECONE_BATCH_SIZE]
            try:
                index.upsert(vectors=batch)
            except Exception as e:
                logger.error(f"Pinecone upsert failed: {e}")

# ------------------------------
# Orchestration
# ------------------------------

def main():
    # 1. Gather all files
    all_pdfs = []
    for category in ["scc", "hcc"]:
        folder = DATA_DIR / category
        if folder.exists():
            all_pdfs.extend(list(folder.glob("*.pdf")))
    all_pdfs.extend(list(DATA_DIR.glob("*.pdf")))
    all_pdfs = list(set(all_pdfs)) # Deduplicate Path objects
    
    # 2. Filter out already processed files
    processed_files = get_processed_files()
    pdfs_to_process = [p for p in all_pdfs if p.name not in processed_files]

    total_files = len(pdfs_to_process)
    skipped_count = len(all_pdfs) - total_files

    logger.info(f"🚀 Found {len(all_pdfs)} PDFs. Skipped {skipped_count}. Processing {total_files} new PDFs.")
    
    if total_files == 0:
        logger.info("Nothing to do.")
        return

    # 3. Process in Memory-Safe Batches
    PDF_PROCESS_BATCH = 50 
    
    with tqdm(total=total_files, desc="Processing PDFs") as pbar:
        for i in range(0, total_files, PDF_PROCESS_BATCH):
            current_pdf_batch = pdfs_to_process[i : i + PDF_PROCESS_BATCH]
            current_filenames = [p.name for p in current_pdf_batch]
            
            # Step A: Parallel Text Extraction
            extracted_chunks_buffer = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS_PDF) as extractor:
                results = extractor.map(extract_text_from_pdf, current_pdf_batch)
                
                for res in results:
                    if res:
                        extracted_chunks_buffer.extend(res)
                    pbar.update(1)

            # Step B: Embed & Upsert
            total_chunks = len(extracted_chunks_buffer)
            if total_chunks > 0:
                chunk_batches = [extracted_chunks_buffer[j:j+PINECONE_BATCH_SIZE] 
                                 for j in range(0, total_chunks, PINECONE_BATCH_SIZE)]
                
                for batch in chunk_batches:
                    process_batch_embeddings_and_upsert(batch)

            # Step C: Mark this batch as processed in JSON
            # We mark them even if they failed extraction (empty chunks) 
            # to prevent infinite retries on corrupted files.
            mark_files_as_processed(current_filenames)

    logger.info("✅ All processing complete.")

if __name__ == "__main__":
    main()
import os
import time
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from pinecone import Pinecone
from google import genai
from tqdm import tqdm

# ------------------------------
# Configuration
# ------------------------------
DATA_DIR = Path(__file__).with_name("bare-acts-json")
PROCESSED_TRACKER_FILE = "processed.json"
INDEX_NAME = "legal-bare-acts-index"

MAX_WORKERS_EMBED = 8
PINECONE_BATCH_SIZE = 50  # Lowered batch size slightly for safety
MAX_METADATA_BYTES = 35000 # Safety limit (Pinecone max is 40KB)

PRINT_EXTRACTED_TEXT = True 

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bare_act_ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------------
# Load ENV
# ------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY or not PINECONE_API_KEY:
    logger.warning("Missing API keys. Embedding will fail, but extraction will work.")
    client = None
    index = None
else:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    client = genai.Client(api_key=GEMINI_API_KEY)

# ------------------------------
# Processed File Tracker
# ------------------------------
def get_processed_files() -> set:
    if os.path.exists(PROCESSED_TRACKER_FILE):
        try:
            with open(PROCESSED_TRACKER_FILE, "r") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def mark_files_as_processed(files: List[str]):
    processed = get_processed_files()
    processed.update(files)
    with open(PROCESSED_TRACKER_FILE, "w") as f:
        json.dump(sorted(processed), f, indent=2)

# ------------------------------
# Helpers: Sanitization & Cleaning
# ------------------------------
def sanitize_id(text: str) -> str:
    """Ensures vector ID is ASCII only and Pinecone compliant."""
    if not text: return "unknown"
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'[^a-zA-Z0-9]', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')[:500]

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def recursive_text_extractor(node: Any) -> str:
    if isinstance(node, str): return node.strip()
    if isinstance(node, dict):
        parts = []
        if "text" in node or "contains" in node:
            if "text" in node: parts.append(recursive_text_extractor(node["text"]))
            if "contains" in node: parts.append(recursive_text_extractor(node["contains"]))
        else:
            try: sorted_keys = sorted(node.keys(), key=lambda x: int(x) if x.isdigit() else 999)
            except: sorted_keys = node.keys()
            for k in sorted_keys: parts.append(recursive_text_extractor(node[k]))
        return "\n".join(part for part in parts if part)
    return ""

def truncate_metadata_text(text: str) -> str:
    """
    Truncates text to ensure metadata JSON stays under Pinecone's 40KB limit.
    """
    encoded = text.encode('utf-8')
    if len(encoded) <= MAX_METADATA_BYTES:
        return text
    
    # If too long, cut it and add a marker
    logger.warning(f"⚠️ Truncating metadata text (Original: {len(encoded)} bytes)")
    truncated = encoded[:MAX_METADATA_BYTES].decode('utf-8', 'ignore')
    return truncated + "... [TRUNCATED DUE TO SIZE LIMIT]"

# ------------------------------
# Bare Act Extraction
# ------------------------------
def extract_sections_from_bare_act(json_path: Path) -> List[Dict[str, Any]]:
    records = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            act = json.load(f)

        act_title = act.get("Act Title", "Unknown Act")
        act_id = act.get("Act ID", "")
        enactment_date = act.get("Enactment Date", "")
        
        year_match = re.search(r"(18|19|20)\d{2}", enactment_date) or re.search(r"(18|19|20)\d{2}", act_id)
        enactment_year = int(year_match.group()) if year_match else 0

        definitions = act.get("Act Definition", {})
        sections = act.get("Sections", {})
        schedules = act.get("Schedule", {})
        footnotes = act.get("Footnotes", {})

        # Flatten footnotes
        flat_footnotes = []
        for page in footnotes.values():
            if isinstance(page, dict): flat_footnotes.extend(page.values())
            elif isinstance(page, str): flat_footnotes.append(page)
        footnote_text = clean_text("\n".join(flat_footnotes))

        # --- PROCESS DEFINITIONS ---
        if definitions:
            definitions_text = recursive_text_extractor(definitions)
            full_def_text = clean_text(f"""
Act Title: {act_title}
Act ID: {act_id}
Definitions / Preamble:
{definitions_text}
""".strip())
            
            records.append({
                "id": sanitize_id(f"{json_path.stem}_definitions"),
                "text": full_def_text,
                "metadata": {
                    "doc_type": "bare_act_definition",
                    "act_title": act_title,
                    "act_id": act_id,
                    "enactment_year": enactment_year,
                    "section_number": "definitions",
                    "jurisdiction": "India",
                    "source": "legislative_text",
                    # Apply truncation here
                    "text": truncate_metadata_text(full_def_text)
                }
            })

        # --- PROCESS SECTIONS ---
        for sec_key, sec in sections.items():
            sec_num_match = re.search(r"Section\s*([\d\w]+)", sec_key, re.IGNORECASE)
            section_number = sec_num_match.group(1) if sec_num_match else sec_key.replace("Section", "").strip(" .")
            heading = sec.get("heading", "")
            section_body = recursive_text_extractor(sec.get("paragraphs", {}))

            full_text = clean_text(f"""
Act Title: {act_title}
Act ID: {act_id}
Enactment Date: {enactment_date}
Section {section_number} – {heading}
{section_body}
Footnotes: {footnote_text if footnote_text else "None"}
""".strip())

            if PRINT_EXTRACTED_TEXT:
                print(f"[SECTION {section_number}] {heading}")

            records.append({
                "id": sanitize_id(f"{json_path.stem}_s{section_number}"),
                "text": full_text,
                "metadata": {
                    "doc_type": "bare_act_section",
                    "act_title": act_title,
                    "act_id": act_id,
                    "enactment_year": enactment_year,
                    "section_number": section_number,
                    "section_heading": heading,
                    "jurisdiction": "India",
                    "source": "legislative_text",
                    # Apply truncation here
                    "text": truncate_metadata_text(full_text)
                }
            })

        # --- PROCESS SCHEDULES ---
        if schedules:
            for sched_key, sched_content in schedules.items():
                schedule_body = recursive_text_extractor(sched_content)
                full_text = clean_text(f"""
Act Title: {act_title}
Act ID: {act_id}
{sched_key}
{schedule_body}
""".strip())
                
                safe_sched_key = sanitize_id(sched_key)
                
                if PRINT_EXTRACTED_TEXT:
                    print(f"[SCHEDULE] {sched_key}")

                records.append({
                    "id": sanitize_id(f"{json_path.stem}_{safe_sched_key}"),
                    "text": full_text,
                    "metadata": {
                        "doc_type": "bare_act_schedule",
                        "act_title": act_title,
                        "act_id": act_id,
                        "enactment_year": enactment_year,
                        "section_number": "Schedule",
                        "section_heading": sched_key,
                        "jurisdiction": "India",
                        "source": "legislative_text",
                        # Apply truncation here (Critical for Schedules)
                        "text": truncate_metadata_text(full_text)
                    }
                })

    except Exception as e:
        logger.error(f"Failed parsing {json_path.name}: {e}", exc_info=True)

    return records

# ------------------------------
# Embedding + Upsert
# ------------------------------
def generate_embedding(text: str, retry=0):
    if not client: return None
    try:
        # Embedding models often have token limits too (e.g. 8k tokens).
        # Ideally we chunk, but for now we rely on the model truncating or handling it.
        # Gemini Text Embedding 004 handles large inputs well.
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=text[:9000] if len(text) > 30000 else text, # Light optimization for embed speed
            config={"task_type": "RETRIEVAL_DOCUMENT"}
        )
        return resp.embeddings[0].values
    except Exception as e:
        if "429" in str(e) and retry < 5:
            time.sleep(2 ** retry)
            return generate_embedding(text, retry + 1)
        logger.error(f"Embedding failed: {e}")
        return None

def embed_and_upsert(records: List[Dict[str, Any]]):
    if not index: return
    
    vectors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_EMBED) as executor:
        # Note: We embed the FULL text (or up to model limit), but metadata is truncated in extraction
        futures = {executor.submit(generate_embedding, r["text"]): r for r in records}

        for future in as_completed(futures):
            record = futures[future]
            embedding = future.result()
            if embedding:
                vectors.append({
                    "id": record["id"],
                    "values": embedding,
                    "metadata": record["metadata"]
                })

    if vectors:
        logger.info(f"Upserting {len(vectors)} vectors...")
        try:
            for i in range(0, len(vectors), PINECONE_BATCH_SIZE):
                batch = vectors[i:i + PINECONE_BATCH_SIZE]
                index.upsert(vectors=batch)
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")

# ------------------------------
# Main
# ------------------------------
def main():
    if not DATA_DIR.exists():
        logger.error(f"Directory {DATA_DIR} not found.")
        return

    files = list(DATA_DIR.glob("*.json"))
    processed = get_processed_files()
    files_to_process = [f for f in files if f.name not in processed]

    logger.info(f"Found {len(files_to_process)} new Bare Act files")
    if not files_to_process: return

    with tqdm(total=len(files_to_process), desc="Ingesting Bare Acts") as pbar:
        for file in files_to_process:
            records = extract_sections_from_bare_act(file)
            if records:
                embed_and_upsert(records)
            mark_files_as_processed([file.name])
            pbar.update(1)

    logger.info("✅ Bare Act ingestion complete")

if __name__ == "__main__":
    main()
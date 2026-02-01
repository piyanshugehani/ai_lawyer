import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Third-party libraries
import pdfplumber
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load Configuration
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
DATA_DIR = Path(__file__).parent / "supreme-court-pdfs-05-06"

# Configs from your .env / previous code
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "models/text-embedding-004" 

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file")

# Initialize Client
client = genai.Client(api_key=GEMINI_API_KEY)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def test_pipeline():
    # 1. Find the first PDF
    all_pdfs = list(DATA_DIR.rglob("*.pdf"))
    if not all_pdfs:
        print(f"No PDFs found in {DATA_DIR}")
        return

    target_pdf = all_pdfs[0] # Just grab the first one
    print(f"--- Processing: {target_pdf.name} ---")

    # 2. Extract Text
    full_text = ""
    with pdfplumber.open(target_pdf) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                full_text += txt + "\n"
    
    print(f"Total Text Length: {len(full_text)} characters")

    # 3. Create Chunks
    chunks = splitter.split_text(full_text)
    print(f"Total Chunks Created: {len(chunks)}")
    print("-" * 40)

    # 4. Generate Embeddings (Test Batch of 3)
    test_batch = chunks[:3] 
    
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=test_batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )

        # 5. Print Results
        for i, (text_chunk, embedding_obj) in enumerate(zip(test_batch, response.embeddings)):
            vector = embedding_obj.values
            
            print(f"\n[Chunk {i+1}]")
            print(f"Text Snippet: {text_chunk[:100]}...") # Print first 100 chars
            print(f"Vector Dimensions: {len(vector)}")     # Should be 768
            print(f"Vector Preview: {vector[:5]}...")     # Print first 5 numbers

    except Exception as e:
        print(f"Error generating embeddings: {e}")

if __name__ == "__main__":
    test_pipeline()
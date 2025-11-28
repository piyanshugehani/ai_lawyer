from pinecone import Pinecone
import os
from pathlib import Path
# ------------------------------
# Load environment variables
# ------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).with_name(".env")) 
except Exception:
    pass

INDEX_NAME = "legal-judgments-index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Delete all vectors in the index
index.delete(delete_all=True)
print("✅ All vectors deleted. Index is now empty.")

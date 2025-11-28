import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # pip install tqdm
from pinecone import Pinecone
from dotenv import load_dotenv

# ------------------------------
# Setup
# ------------------------------
try:
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
except Exception:
    pass

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "legal-judgments-index"

if not PINECONE_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY")

# Initialize
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ------------------------------
# Configuration
# ------------------------------
TARGET_CATEGORY = "scc"  # The category value you want to set for ALL vectors
MAX_WORKERS = 30         # Parallel threads (Don't go too high to avoid Rate Limits)

def update_single_vector(vector_id):
    """
    Updates the metadata for a single vector ID.
    """
    try:
        index.update(
            id=vector_id,
            set_metadata={"category": TARGET_CATEGORY}
        )
        return True
    except Exception as e:
        # If rate limited, we could add retry logic here, 
        # but usually the thread pool handles it via sheer volume.
        print(f"❌ Error updating {vector_id}: {e}")
        return False

def main():
    print(f"🚀 Starting Metadata Update for Index: {INDEX_NAME}")
    print(f"🎯 Setting ALL 'category' fields to: '{TARGET_CATEGORY}'")
    
    # 1. Fetch all IDs (This might take a moment for 60k+ records)
    print("📋 Listing all vector IDs...")
    try:
        # .list() returns a generator that yields IDs
        # We convert to list to measure length for the progress bar
        all_vector_ids = list(index.list())
    except Exception as e:
        print(f"Error listing vectors: {e}")
        return

    total_vectors = len(all_vector_ids)
    print(f"✅ Found {total_vectors} vectors.")

    if total_vectors == 0:
        print("Nothing to update.")
        return

    print("⚡ Starting parallel updates...")
    
    # 2. Update in Parallel
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(update_single_vector, vid): vid for vid in all_vector_ids}
        
        # Track progress
        with tqdm(total=total_vectors, desc="Updating Metadata") as pbar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                pbar.update(1)

    print("\n" + "="*40)
    print(f"🎉 COMPLETED.")
    print(f"✅ Successfully updated: {success_count}/{total_vectors}")
    print(f"ℹ️  All vectors now have category='{TARGET_CATEGORY}'")
    print("="*40)

if __name__ == "__main__":
    main()
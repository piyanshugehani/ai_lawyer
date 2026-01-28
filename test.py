import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# 1. PRINT LIBRARY VERSION
print(f"🔍 Google Library Version: {genai.__version__}")
# If this prints 0.3.x or 0.4.x, you MUST upgrade. 
# It needs to be 0.7.0 or higher.

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print(f"🔑 Testing API Key: {api_key[:5]}...{api_key[-5:] if api_key else 'None'}")

if not api_key:
    print("❌ API Key not found in .env")
    exit()

genai.configure(api_key=api_key)

# 2. LIST MODELS (Safe Mode)
print("\n📋 Scanning Available Models...")
try:
    found_models = []
    for m in genai.list_models():
        if 'embedContent' in m.supported_generation_methods:
            found_models.append(m.name)
            print(f" - {m.name}")
            
    if not found_models:
        print("⚠️ No embedding models found for this key.")
        
except TypeError as e:
    print(f"❌ CRITICAL LIBRARY ERROR: {e}")
    print("👉 CAUSE: Your 'google-generativeai' library is too old.")
    print("👉 FIX: Run: pip install google-generativeai --upgrade --force-reinstall")

# 3. TEST EMBEDDING
print("\n🧪 Testing Embedding Generation...")
try:
    # Try the modern model first
    model_name = "models/text-embedding-004"
    print(f"   Attempting with: {model_name}")
    
    result = genai.embed_content(
        model=model_name,
        content="Hello world",
        task_type="retrieval_document"
    )
    print(f"✅ Success! Generated embedding with length: {len(result['embedding'])}")
    
except Exception as e:
    print(f"⚠️ Failed with {model_name}: {e}")
    
    # Fallback to older model
    print("\n   Attempting fallback with: models/embedding-001")
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content="Hello world",
            task_type="retrieval_document"
        )
        print(f"✅ Success! Fallback worked. Length: {len(result['embedding'])}")
    except Exception as e2:
        print(f"❌ Fallback Failed. Error: {e2}")
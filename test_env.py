# test_env.py
try:
    from google import genai
    print("✅ google-genai is installed correctly!")
except ImportError as e:
    print(f"❌ Still failing: {e}")
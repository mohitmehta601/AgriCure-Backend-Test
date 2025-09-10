#!/usr/bin/env python3
"""
Test Gemini API key configuration
"""
import os
from dotenv import load_dotenv

def test_gemini_api():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found!")
        print("Please add it to the .env file or set as environment variable")
        return False
    
    if api_key == "your_actual_api_key_here":
        print("❌ Please replace 'your_actual_api_key_here' with your real API key")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-10:]}")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("Say 'Hello, API is working!'")
        print(f"✅ Gemini API Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ API Test Failed: {e}")
        return False

if __name__ == "__main__":
    test_gemini_api()

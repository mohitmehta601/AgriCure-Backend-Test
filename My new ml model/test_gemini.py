import google.generativeai as genai

# Configure with your API key
genai.configure(api_key='AIzaSyDKYRhXUSFVgy1LTIXb-G1-syiZwBwA1ps')

# List available models
print("Available Gemini models:")
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"Supports: {model.supported_generation_methods}")
    print()

# Test with a known model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content('Hello, this is a test.')
    print("API test successful!")
    print("Response:", response.text[:100])
except Exception as e:
    print(f"Error with gemini-1.5-flash: {e}")
    
    # Try another model
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content('Hello, this is a test.')
        print("API test successful with gemini-1.5-pro!")
        print("Response:", response.text[:100])
    except Exception as e:
        print(f"Error with gemini-1.5-pro: {e}")

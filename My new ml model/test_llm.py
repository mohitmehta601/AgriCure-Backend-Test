from llm import generate_recommendation_report

# Test data
base_inputs = {
    "Sowing_Date": "2024-03-15",
    "Field_Size": 5.0,
    "Field_Unit": "hectares",
    "N": 50,
    "P": 30,
    "K": 40,
    "pH": 6.5
}

predictions = {
    "Primary_Fertilizer": "NPK 15-15-15",
    "Secondary_Fertilizer": "Urea",
    "pH_Amendment": "None"
}

confidences = {
    "Primary_Fertilizer": 0.85,
    "Secondary_Fertilizer": 0.78,
    "pH_Amendment": 0.92
}

print("Testing Gemini API with recommendation generation...")
try:
    result = generate_recommendation_report(base_inputs, predictions, confidences)
    print("SUCCESS! Generated recommendation:")
    print(f"Keys in result: {list(result.keys())}")
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Sample output:", str(result)[:200], "...")
except Exception as e:
    print(f"ERROR: {e}")

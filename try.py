import google.generativeai as genai

genai.configure(api_key="AIzaSyANmqxIG22VDP2lLV9FoynMy7_R5KQMJJ0")

def list_models():
    try:
        models = genai.list_models()
        for model in models:
            print(f"Model: {model.name}")
    except Exception as e:
        print(f"Error fetching models: {e}")

list_models()

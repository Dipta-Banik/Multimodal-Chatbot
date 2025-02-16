import google.generativeai as genai
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY is missing. Please check your .env file.")

genai.configure(api_key=api_key)


generation_config = {
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 64,
    "max_output_tokens": 8192,
}


model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def translate_text(text, source_language, target_language, retries=10):
    prompt = f"Translate the following text from {source_language} to {target_language}: \"{text}\""

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "Resource has been exhausted" in error_msg:
                wait_time = (attempt + 1) * 2 
                print(f"⚠️ Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return f"Error: {error_msg}"
    
    return "Error: Failed to translate text after multiple attempts. Please try again later."



'''
source_text = input("Enter the text to translate: ")
source_lang = input("Source Language (e.g., English, French, etc.): ")
target_lang = input("Target Language (e.g., English, French, etc.): ")

translated_text = translate_text(source_text, source_lang, target_lang)
print("\nTranslated Text:", translated_text)
'''
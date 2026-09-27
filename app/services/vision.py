import os
import time
from PIL import Image
from google import genai
from google.genai import types

from app.config import settings
from app.schema import PhotoTagExtraction

client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None

def analyze_product_image(image_path: str) -> tuple[PhotoTagExtraction, dict]:
    if not client:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env file.")

    clean_path = os.path.abspath(image_path.replace("\\", "/"))

    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Image file not found at path: {clean_path}")

    img = Image.open("D:/flyrank ai/ecommerce catalog/image data/bag 9.jpg")

    prompt = (
        "You are an expert e-commerce catalog inspector. "
        "Analyze this product image and extract structured attributes matching the schema. "
        "Be accurate with color, material, category, and self-assess your confidence."
    )

    # Active Gemini Flash models supported by the API
    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
    response = None
    last_error = None

    for model_name in candidate_models:
        try:
            print(f"Sending image to model: {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=[img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PhotoTagExtraction,
                    temperature=0.1,
                ),
            )
            print(f"Successfully processed image using model: {model_name}")
            break  # Success! Exit loop
        except Exception as e:
            last_error = e
            err_msg = str(e)
            if "503" in err_msg or "UNAVAILABLE" in err_msg or "404" in err_msg:
                print(f"Model {model_name} temporary error/unavailable. Retrying next active model...")
                time.sleep(1)
            else:
                raise e

    if not response:
        raise RuntimeError(f"All candidate models failed. Last error: {last_error}")

    extracted_tags = PhotoTagExtraction.model_validate_json(response.text)

    usage = response.usage_metadata
    metrics = {
        "prompt_tokens": usage.prompt_token_count if usage else 0,
        "completion_tokens": usage.candidates_token_count if usage else 0,
        "estimated_cost_usd": 0.0
    }

    return extracted_tags, metrics
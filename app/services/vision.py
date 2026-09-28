import os
import base64
import json
import re
from openai import OpenAI
from app.config import settings
from app.schema import PhotoTagExtraction

api_key = getattr(settings, "OPENROUTER_API_KEY", None) or os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key if api_key else "placeholder"
)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def clean_and_parse_json(text: str) -> dict:
    if not text:
        return {}

    # Strip markdown formatting or safety headers
    text = re.sub(r"User Safety:\s*\w+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}

def analyze_product_image(image_path: str) -> tuple[PhotoTagExtraction, dict]:
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    clean_path = os.path.abspath(image_path.replace("\\", "/"))
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Image file not found: {clean_path}")

    base64_image = encode_image_to_base64(clean_path)
    mime_type = "image/png" if clean_path.lower().endswith(".png") else "image/jpeg"

    prompt = (
        "Look at this e-commerce product image and describe it in detail. "
        "Return ONLY a JSON object with keys: subject, category, color, material, confidence, caption. "
        'Example format: {"subject": "travel bag", "category": "bags", "color": "blue", "material": "canvas", "confidence": 0.95, "caption": "A blue canvas backpack with zippered pockets."}'
    )

    # Active free multimodal vision endpoint on OpenRouter
    target_model = "stealth/space-bunny-alpha"

    response = None
    data = {}

    try:
        print(f"\n[VISION] Calling OpenRouter model: {target_model} for {os.path.basename(clean_path)}...")
        response = client.chat.completions.create(
            model=target_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0.1
        )

        if response.choices and response.choices[0].message.content:
            raw_text = response.choices[0].message.content
            print(f"--- RAW RESPONSE FROM SPACE BUNNY ALPHA ---")
            print(raw_text[:300])
            print("------------------------------------------")

            data = clean_and_parse_json(raw_text)

    except Exception as e:
        print(f"  ❌ API Error for {os.path.basename(clean_path)}: {e}")

    if not data or not data.get("subject"):
        print(f"  ⚠ Model failed to produce valid JSON tags.")
        data = {
            "subject": "product item",
            "category": "general",
            "color": "unknown",
            "material": "unknown",
            "caption": "E-commerce catalog photo",
            "confidence": 0.50
        }

    extracted_tags = PhotoTagExtraction(
        subject=data.get("subject", "product"),
        category=data.get("category", "general"),
        color=data.get("color", "unknown"),
        material=data.get("material", "unknown"),
        attributes_json={"source": f"openrouter-{target_model}"},
        caption=data.get("caption", "Product item"),
        confidence=float(data.get("confidence", 0.85))
    )

    metrics = {
        "prompt_tokens": response.usage.prompt_tokens if response and hasattr(response, "usage") and response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response and hasattr(response, "usage") and response.usage else 0,
        "estimated_cost_usd": 0.0
    }

    return extracted_tags, metrics
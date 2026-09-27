from typing import List, Optional
from pydantic import BaseModel, Field

class PhotoTagExtraction(BaseModel):
    subject: str = Field(
        description="Primary object detected, e.g, 'jacket', 'sneaker'"
    )
    
    category: str = Field(
        description="Top level Taxonomy, e.g, 'Accessory', 'Footwear'"
    )
    
    color: str = Field(
        description="Primary color, e.g, 'red', 'blue'"
    )
    
    material: str = Field(
        description="Stuff of the product, e.g, 'leather', 'cotton'"
    )
    
    attributes: List[str] = Field(
        default_factory=list,
        description="List of visual features, e.g. ['bifold', 'stitched']"
    )
    
    caption: str = Field(
        description="Objective one-sentence summary of what is visible in the photo"
    )
    
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model self-assessed certainty score between 0.0 and 1.0"
    )

# --- Add these two classes to the bottom of your app/schema.py ---

class PhotoIngestRequest(BaseModel):
    file_path: str = Field(
        description="Local path to photo file, e.g. 'data/images/wallet_red_01.jpg'"
    )

class PhotoIngestResponse(BaseModel):
    photo_id: str
    file_path: str
    status: str
    extracted_tags: Optional[PhotoTagExtraction] = None
    reason: Optional[str] = None
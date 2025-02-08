from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class BoundingBox(BaseModel):
    """Model for bounding box coordinates."""
    x1: float = Field(..., ge=0.0, le=1.0)  # Must be between 0 and 1
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)

class Metadata(BaseModel):
    title: str
    pages: int
    parser: str
    text_nodes: List[str]
    bounding_boxes: Dict[str, BoundingBox] = {}  # Using our BoundingBox model

class ParsedPDF(BaseModel):
    metadata: Metadata
    content: str
from pydantic import BaseModel, Field
from typing import List


class ProfileEmbedding(BaseModel):
    """Represents a profile with its embedding vector."""
    profile_id: str = Field(..., description="Unique identifier for the profile")
    embedding: List[float] = Field(..., description="Vector representation of the profile's preferences")

    class Config:
        json_schema_extra = {
            "example": {
                "profile_id": "user123_action_movies",
                "embedding": [0.1, 0.2, 0.3, 0.4]
            }
        }


class ProfileIngestionRequest(BaseModel):
    """Request payload for ingesting media and computing profile embeddings."""
    image_urls: List[str] = Field(..., description="List of image URLs to process for embedding computation")

    class Config:
        json_schema_extra = {
            "example": {
                "image_urls": [
                    "https://example.com/movie1.jpg",
                    "https://example.com/movie2.jpg"
                ]
            }
        }


class ProfileIngestionResponse(BaseModel):
    """Response after processing profile ingestion."""
    message: str = Field(..., description="Status message")

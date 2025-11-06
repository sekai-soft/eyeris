from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from eyeris.models import ProfileIngestionRequest, ProfileIngestionResponse
from eyeris.embeddings import compute_embeddings_from_images, store_profile_embeddings, warmup_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler - runs on startup and shutdown."""
    # Startup: Pre-load the ResNet-50 model
    warmup_model()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Eyeris",
    description="Simple media recommendation engine",
    version="0.1.0",
    lifespan=lifespan
)


@app.get("/")
def hello_world():
    return {"message": "Hello World"}


@app.post("/profile/{profile_id}/ingest", response_model=ProfileIngestionResponse)
async def ingest_profile_media(profile_id: str, request: ProfileIngestionRequest):
    """
    Ingest media for a profile and compute/store embeddings.

    This endpoint accepts a list of image URLs, processes them to extract embeddings,
    and stores them for the profile.
    """
    try:
        embeddings = await compute_embeddings_from_images(request.image_urls)
        await store_profile_embeddings(profile_id, embeddings)

        return ProfileIngestionResponse(
            message="Successfully processed and stored profile embeddings"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing profile: {str(e)}")

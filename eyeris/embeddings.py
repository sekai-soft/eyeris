from typing import List, Tuple
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import httpx
from io import BytesIO
import ssl
import certifi


# Load ResNet-50 model and remove the final classification layer
_resnet_model = None
_transform = None
_device = None


def _get_model_and_transform():
    """Lazy load the ResNet-50 model and transforms."""
    global _resnet_model, _transform, _device

    if _resnet_model is None:
        # Auto-detect device (use GPU if available)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set up SSL context with certifi for model downloads
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl._create_default_https_context = lambda: ssl_context

        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove the final classification layer to get embeddings
        _resnet_model = torch.nn.Sequential(*list(resnet.children())[:-1])
        _resnet_model.eval()  # Set to evaluation mode
        _resnet_model = _resnet_model.to(_device)

        # Standard ImageNet preprocessing
        _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    return _resnet_model, _transform


def warmup_model():
    """
    Pre-load the ResNet-50 model and download weights if needed.

    Call this on app startup to avoid cold start delays on first request.
    """
    _get_model_and_transform()


async def compute_embeddings_from_images(image_urls: List[str]) -> List[Tuple[str, List[float]]]:
    """
    Compute embeddings from a list of image URLs using ResNet-50.

    Args:
        image_urls: List of URLs pointing to images to process

    Returns:
        List of tuples (image_url, embedding_vector) where embedding_vector is 2048-dimensional

    Raises:
        Exception: If image download or processing fails
    """
    model, transform = _get_model_and_transform()
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for url in image_urls:
            # Download image
            response = await client.get(url)
            response.raise_for_status()

            # Load and preprocess image
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(_device)  # Add batch dimension and move to device

            # Extract embedding
            with torch.no_grad():
                embedding = model(image_tensor)
                # Flatten to 1D vector (batch_size, 2048, 1, 1) -> (2048,)
                embedding = embedding.squeeze().cpu().numpy()
                results.append((url, embedding.tolist()))

    return results


async def store_profile_embeddings(profile_id: str, embeddings_with_urls: List[Tuple[str, List[float]]]) -> None:
    """
    Store new embeddings for a profile.

    TODO: Implement storage logic to persist embeddings to database/storage.
    This will store all individual embeddings for the profile, not merge them.

    Args:
        profile_id: Unique identifier for the profile
        embeddings_with_urls: List of tuples (image_url, embedding_vector) to store
    """
    # Placeholder: Add storage implementation here
    pass

import requests
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import torch
from transformers import ViTImageProcessor, ViTModel

# Initialize the Vision Transformer model and processor
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    lambda image: image.convert("RGB"),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def process_image(image_bytes):
    """Apply transformations to an image."""
    with Image.open(BytesIO(image_bytes)) as img:
        return transform(img).unsqueeze(0)

def get_image_embeddings(image_tensor):
    """Generate embeddings for an image tensor using the pre-trained model."""
    if torch.cuda.is_available():
        image_tensor = image_tensor.to('cuda')
        model.to('cuda')

    with torch.no_grad():  # Disable gradient calculations
        outputs = model(image_tensor)
        embeddings = outputs.last_hidden_state.cpu().numpy()  # Extract embeddings and move to CPU
    return embeddings.tolist()

def embed_image(image_url):
    """Fetch an image from a URL and get its embeddings."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise exception for HTTP errors

        image_tensor = process_image(response.content)
        embeddings_list = get_image_embeddings(image_tensor)
        return embeddings_list

    except requests.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return "Fetch the image failed"
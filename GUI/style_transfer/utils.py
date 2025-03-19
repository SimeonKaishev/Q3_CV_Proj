import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import random

def load_image(image_input, max_size=512):
    """Loads an image from a file path or NumPy array, ensures it’s RGB, resizes, and converts to a tensor."""
    
    if isinstance(image_input, str):  # ✅ If it's a file path, open it normally
        image = Image.open(image_input).convert("RGB")

    elif isinstance(image_input, np.ndarray):  # ✅ If it's a NumPy array (from OpenCV)
        image = Image.fromarray(image_input).convert("RGB")

    else:
        raise TypeError("Invalid image format! Expected file path or NumPy array.")

    # Resize if too large
    size = max(image.size)
    if size > max_size:
        size = max_size

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def select_style_images(mood, num_styles=3):
    """Selects multiple style images from the Style Bank based on mood."""
    style_bank_dir = "GUI/style_transfer/style_bank"
    available_styles = [f for f in os.listdir(style_bank_dir) if mood.lower() in f.lower()]
    
    # Select 3 random style images
    selected_styles = random.sample(available_styles, min(num_styles, len(available_styles)))
    
    return [os.path.join(style_bank_dir, style) for style in selected_styles]

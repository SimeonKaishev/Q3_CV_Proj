# image_generator.py

import torch
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image

# Load the model once (avoiding reloading on each function call)
model_id = "stabilityai/stable-diffusion-3.5-medium"
pipeline = DiffusionPipeline.from_pretrained(model_id)
#pipeline.to("cuda")

def generate_image(prompt, steps=5, guidance=7.5):
    """Generates an image and returns it as a NumPy array."""
    print("generating image")
    image = pipeline(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]

    print("converting to np array")
    # Convert PIL Image to NumPy array (H, W, C) format
    image_np = np.array(image)

    print("done")
    return image_np  # Return NumPy array
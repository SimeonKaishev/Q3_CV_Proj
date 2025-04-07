import torch
import numpy as np
from diffusers import AutoPipelineForText2Image
from diffusers import DiffusionPipeline
from PIL import Image

# Global variable to hold the model
pipeline = None


def load_model():
    """Loads the SD model with LoRA weights once."""
    global pipeline
    if pipeline is None:
        print("Loading model...")
        pipeline = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
        pipeline.load_lora_weights("sskaishev/album_art")

        print("Model loaded with LoRA weights.")
    return pipeline


def generate_image(prompt, steps=25, guidance=7.5, height=512, width=512):
    """Generates an image using the loaded model."""
    global pipeline
    if pipeline is None:
        raise ValueError("Model is not loaded. Call load_model() first.")

    print(f"Generating image for prompt: {prompt}")
    result = pipeline(
        prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width
    )
    print(f"num images: {result.images}")
    image = result.images[0]
    image_np = np.array(image)
    return image_np
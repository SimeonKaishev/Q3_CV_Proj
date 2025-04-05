import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from diffusers import StableDiffusionImg2ImgPipeline

class StableDiffusionPhotoStyle:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        # Load the model (will download from Hugging Face if not cached)
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.pipe.to(device)
        self.device = device

    def transfer(
        self,
        content_img: Image.Image,
        style_prompt: str,
        strength: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ) -> Image.Image:
        """
        :param content_img: The original photo (PIL Image)
        :param style_prompt: Text prompt describing the style (e.g. "photorealistic image of a leather-clad fashion shoot, bright colors, vibrant lights")
        :param strength: How strongly to transform (0.0 -> minimal change, 1.0 -> max change)
        :param guidance_scale:  Higher = follow prompt more strictly
        :param num_inference_steps:  More steps = better quality, slower
        :return: PIL Image with style applied
        """
        # Run img2img
        result = self.pipe(
            prompt=style_prompt,
            image=content_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        return result.images[0]  # The pipeline returns [list_of_images]


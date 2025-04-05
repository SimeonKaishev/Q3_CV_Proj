import streamlit as st
from PIL import Image
import os
import sys

# Get the absolute path of the Scripts directory
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Add Scripts/ to the system path
sys.path.append(script_dir)


from sd_helper import load_model, generate_image
##import scripts.sd_helper as ig
import torch

st.set_page_config(page_title="LoRA Image Generator", layout="centered")

# Load model once
@st.cache_resource
def initialize_model():
    return load_model(
        base_model="stabilityai/stable-diffusion-2-1",
        lora_path="path/to/lora/dir",
        weight_name="pytorch_lora_weights.safetensors"
    )

st.title("🎨 LoRA Fine-Tuned Image Generator")
model = initialize_model()

prompt = st.text_input("Enter your prompt:", "A vaporwave album cover with neon colors")
steps = st.slider("Steps", 5, 50, 25)
guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

if st.button("Generate"):
    with st.spinner("Generating..."):
        image_np = generate_image(prompt, steps=steps, guidance=guidance)
        st.image(Image.fromarray(image_np), caption="Generated Image", use_column_width=True)
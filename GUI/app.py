import sys
import os
import asyncio
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from style_transfer.utils import load_image, select_style_images
from style_transfer.sd_style import StableDiffusionPhotoStyle
# Get the absolute path of the Scripts directory
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Add Scripts/ to the system path
sys.path.append(script_dir)


from sd_helper import load_model, generate_image
##import scripts.sd_helper as ig
import torch



if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load model once
@st.cache_resource
def initialize_model():
    return load_model(
        base_model="stabilityai/stable-diffusion-2-1",
        lora_path="path/to/lora/dir",
        weight_name="pytorch_lora_weights.safetensors"
    )

st.markdown(
    "<h1 style='text-align: center;'>🎨 Album Art Generator</h1>",
    unsafe_allow_html=True
)

model = initialize_model()

# Centered intro message
st.markdown(
    "<div style='text-align: center;'>"
    "<h4>Simply fill out the form below and let the AI do the rest!</h4>"
    "</div>",
    unsafe_allow_html=True
)

# Artist & Album Info
artist_name = st.text_input("🎤 Artist Name")
album_name = st.text_input("💿 Album Name")

# Session state for images
if "image" not in st.session_state:
    st.session_state["image"] = None
if "stylized_images" not in st.session_state:
    st.session_state["stylized_images"] = []

# Main logic starts when both artist and album name are filled
if artist_name and album_name:

    # Choice: Generate or Upload
    image_choice = st.radio(
        "How would you like to start?",
        ("Generate an image", "Upload an image")
    )

    if image_choice == "Generate an image":
        # Show form fields specific to generation
        lyrics = st.text_area("🎵 Key Lyrics (Optional)")
        genre = st.selectbox(
            "🎧 Select Music Genre",
            ["Pop", "Rock", "Hip-Hop", "Jazz", "Electronic", "Classical", "Country", "Indie", "R&B", "Metal"]
        )
        instructions = st.text_area("📝 Additional Style Instructions (Optional)")
        resolution = st.selectbox("🖼️ Desired Resolution", ["512x512", "768x768", "1024x1024"])

        steps = st.slider("Steps", 5, 50, 25)
        guidance = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

        if st.button("🎨 Generate Image"):
            #st.write(f"Generating image for **{artist_name} – {album_name}**...")

            # Construct prompt from user input
            prompt = f"An album cover for a {genre} album titled '{album_name}' by {artist_name}."
            if lyrics:
                prompt += f" Inspired by the lyrics: '{lyrics.strip()}'"
            if instructions:
                prompt += f" Style notes: {instructions.strip()}"

            # Placeholder for image generation
            #model = StableDiffusionPhotoStyle()
            #generated_image = model.generate(prompt, resolution=resolution)
            with st.spinner("Generating..."):
                image_np = generate_image(prompt, steps=steps, guidance=guidance)

            st.session_state["image"] = image_np

            st.image(image_np, caption="Generated Album Cover", use_container_width=True)

    elif image_choice == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.session_state["image"] = np.array(uploaded_image)
            st.image(st.session_state["image"], caption="Uploaded Image", use_container_width=True)

    # Final image selection
    if st.session_state["image"] is not None:
        st.write("### ✅ Final Album Cover")
        st.image(st.session_state["image"], caption="Selected Final Art", use_container_width=True)

else:
    st.warning("⛔ Please enter both artist and album name to continue.")

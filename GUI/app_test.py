import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

# Get the absolute path of the Scripts directory
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Add Scripts/ to the system path
sys.path.append(script_dir)

from generate_art import generate_image






# Title of the app
st.title("Album Art Generator")

# Step 1: Ask for artist name, album name, and optional inputs
artist_name = st.text_input("Artist Name")
album_name = st.text_input("Album Name")
lyrics = st.text_area("Lyrics (Optional)")
instructions = st.text_area("Instructions (Optional)")

# Store images in session state
if "image" not in st.session_state:
    st.session_state["image"] = None

if artist_name and album_name:
    # Step 2: Allow user to either generate an image or upload one
    image_choice = st.radio(
        "What would you like to do next?",
        ("Generate an image", "Upload an image")
    )

    if image_choice == "Generate an image":
        if st.button("Generate Image"):
            st.write(f"Generating an image for artist: {artist_name}, album: {album_name}")
            prompt = f"Album cover for {artist_name} - {album_name}. {instructions}"
            st.session_state["image"] = generate_image(prompt)
            st.image(Image.fromarray(st.session_state["image"]), caption="Generated Image", use_container_width=True)

    elif image_choice == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            st.session_state["image"] = np.array(uploaded_image)
            st.image(st.session_state["image"], caption="Uploaded Image", use_container_width=True)

else:
    st.warning("Please enter both artist and album name to proceed.")
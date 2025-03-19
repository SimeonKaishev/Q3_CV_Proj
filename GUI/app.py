import sys
import os
import asyncio
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
from style_transfer.utils import load_image, select_style_images
from style_transfer.model import StyleTransferModel

# ✅ Fix for Windows event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
if "stylized_images" not in st.session_state:
    st.session_state["stylized_images"] = []

if artist_name and album_name:
    # Step 2: Allow user to either generate an image or upload one
    image_choice = st.radio(
        "What would you like to do next?",
        ("Generate an image", "Upload an image")
    )

    if image_choice == "Generate an image":
        if st.button("Generate Image"):
            st.write(f"Generating an image for artist: {artist_name}, album: {album_name}")

            # Placeholder image (User would replace this with actual AI-generated image)
            generated_img = np.zeros((300, 300, 3), dtype=np.uint8)
            generated_img = cv2.putText(
                generated_img,
                f"{artist_name} - {album_name}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            st.session_state["image"] = generated_img
            st.image(st.session_state["image"], caption="Generated Image", use_container_width=True)

    elif image_choice == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            uploaded_image = np.array(uploaded_image)
            st.session_state["image"] = uploaded_image
            st.image(st.session_state["image"], caption="Uploaded Image", use_container_width=True)

    # Step 3: Mood selection and stylization
    if st.session_state["image"] is not None:
        mood = st.selectbox("Select the mood for the album cover", ["Happy", "Sad", "Exciting", "Relaxed"])

        if st.button("Stylize Image"):
            st.write(f"Applying {mood} style transfer...")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            content_img = load_image(st.session_state["image"]).to(device)

            # Select 3 random style images based on mood
            style_images = select_style_images(mood, num_styles=3)

            stylized_versions = []
            for style_path in style_images:
                style_img = load_image(style_path).to(device)
                model = StyleTransferModel(content_img, style_img)
                
                # Generate 3 variations per style image
                for _ in range(3):
                    output_img = model.train(num_steps=30)
                    output_img = output_img.squeeze().permute(1, 2, 0).cpu().detach().numpy()
                    output_img = np.clip(output_img, 0, 1)  # ✅ Fix: Ensure values are in range [0,1]
                    stylized_versions.append(output_img)

            # Save stylized images to session state
            st.session_state["stylized_images"] = stylized_versions

        # Display 3x3 grid of stylized images
        if st.session_state["stylized_images"]:
            st.write("### Select Your Favorite Style:")
            cols = st.columns(3)
            for i, img in enumerate(st.session_state["stylized_images"]):
                img = np.clip(img, 0, 1)  # ✅ Fix: Ensure images are valid
                with cols[i % 3]:  
                    st.image(img, use_container_width=True, caption=f"Style Option {i+1}")
                    if st.button(f"Select Style {i+1}", key=f"style_{i}"):
                        st.session_state["final_choice"] = img

        # Show final selected image
        if "final_choice" in st.session_state:
            st.write("### Selected Final Cover")
            st.image(st.session_state["final_choice"], caption="Final Album Cover", use_container_width=True)

else:
    st.warning("Please enter both artist and album name to proceed.")

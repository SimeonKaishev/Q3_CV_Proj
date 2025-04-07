import sys
import os
import asyncio
import streamlit as st
import cv2
import numpy as np

import torch
from PIL import Image

#add scripts/ to path
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.append(script_dir)


from sd_helper import load_model, generate_image
from title_placement import place_text_using_visual_balance


##import scripts.sd_helper as ig
import torch



if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# initialise sd model
@st.cache_resource
def initialize_model():
    return load_model()

st.markdown(
    "<h1 style='text-align: center;'>🎨 Album Art Generator</h1>",
    unsafe_allow_html=True
)
model = initialize_model()


st.markdown(
    "<div style='text-align: center;'>"
    "<h4>Simply fill out the form below and let the AI do the rest!</h4>"
    "</div>",
    unsafe_allow_html=True
)

# initial input
artist_name = st.text_input("🎤 Artist Name")
album_name = st.text_input("💿 Album Name")

# session state for images
if "image" not in st.session_state:
    st.session_state["image"] = None
if "stylized_images" not in st.session_state:
    st.session_state["stylized_images"] = []

#main logic
if artist_name and album_name:
    # ------------- Image Gen Stuff -------------

    # choose between generating or uploading album image
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

        grid_option = st.selectbox("How many images to generate?", [1, 3, 9])

        if st.button("🎨 Generate Image"):
            #construct prompt from user input
            prompt = f"An album cover for a {genre} album titled '{album_name}' by {artist_name}."
            if lyrics:
                prompt += f" Inspired by the lyrics: '{lyrics.strip()}'"
            if instructions:
                prompt += f" Style notes: {instructions.strip()}"

            stylized_versions = []
            progress_bar = st.progress(0)

            #generate images
            for i in range(grid_option):
                with st.spinner(f"Generating image {i+1}/{grid_option}..."):
                    img = generate_image(prompt, steps=steps, guidance=guidance)
                    stylized_versions.append(img)
                    progress_bar.progress((i + 1) / grid_option)

            st.session_state["stylized_images"] = stylized_versions
            #reset choice
            st.session_state["image"] = None  

        #display generated images
        if st.session_state.get("stylized_images"):
            st.markdown("### Choose One of the Generated Images")

            cols = st.columns(3)
            for idx, img in enumerate(st.session_state["stylized_images"]):
                with cols[idx % 3]:
                    st.image(img, use_container_width=True, caption=f"Option {idx + 1}")
                    if st.button(f"Select Option {idx + 1}", key=f"select_{idx}"):
                        st.session_state["image"] = img
                        st.success(f"Selected Option {idx + 1}")

            #show images
            cols = st.columns(grid_option if grid_option <= 3 else 3)

    elif image_choice == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            st.session_state["image"] = np.array(uploaded_image)
            st.image(st.session_state["image"], caption="Uploaded Image", use_container_width=True)

    # show selected image
    if st.session_state["image"] is not None:
        st.write("###Final Album Cover")
        st.image(st.session_state["image"], caption="Selected Final Art", use_container_width=True)

        # ------------- Title Stuff -------------
        st.divider()
        st.header("📝 Add Title to Album Cover")

        # open cv fonts
        font_options = {
            "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
            "Complex": cv2.FONT_HERSHEY_COMPLEX,
            "Duplex": cv2.FONT_HERSHEY_DUPLEX,
            "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
            "Simplex Script": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            "Complex Script": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
            "Plain": cv2.FONT_HERSHEY_PLAIN,
        }

        #user inputs
        text_size_option = st.selectbox("Select Text Size", ["Big", "Medium", "Small"])
        image_np = np.array(st.session_state["image"])
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image_cv.shape

        # determine text size in pixels 
        if text_size_option == "Big":
            text_size = int(image_width * 0.75), 60
        elif text_size_option == "Medium":
            text_size = int(image_width * 0.50), 60
        else:
            text_size = int(image_width * 0.25), 60

        # get inputs
        selected_font = st.selectbox("Select Font", list(font_options.keys()))
        font = font_options[selected_font]
        color_hex = st.color_picker("Pick Text Color", "#FFFFFF")
        text_color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        thickness = st.slider("Text Thickness", min_value=1, max_value=10, value=2)

        title_text = st.text_input("Title Text", value=f"{album_name} - {artist_name}")

        #place title
        if st.button("Place Title"):
            # determine title position
            position = place_text_using_visual_balance(image_cv, text_size=text_size, use_diagonal=True)
            x, y = position if position else (100, 100)

            #save state
            st.session_state.title = title_text
            st.session_state.font = font
            st.session_state.text_size = text_size
            st.session_state.font_color = text_color
            st.session_state.font_thickness = thickness
            st.session_state.font_name = selected_font
            st.session_state.x = x
            st.session_state.y = y
            st.session_state.placed = True

        if st.session_state.get("placed", False):
            #add sliders to adjust position
            st.header("🎛️ Adjust Title Position")
            x_offset = st.slider("X Offset", min_value=0, max_value=image_width, value=st.session_state.x)
            y_offset = st.slider("Y Offset", min_value=0, max_value=image_height, value=st.session_state.y)

            # func to change font scale so text fits in the dims determined by title size
            def calculate_font_scale(text, max_width, max_height, font, thickness):
                font_scale = 1.0
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                if tw < max_width:
                    while tw < max_width and th < max_height:
                        font_scale += 0.1
                        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                else:
                    while tw > max_width:
                        font_scale -= 0.1
                        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                return font_scale

            font_scale = calculate_font_scale(
                st.session_state.title,
                st.session_state.text_size[0],
                st.session_state.text_size[1],
                font=st.session_state.font,
                thickness=st.session_state.font_thickness
            )
            (text_w, text_h), _ = cv2.getTextSize(st.session_state.title, st.session_state.font, font_scale, st.session_state.font_thickness)
            
            #get final text pos
            x_text = x_offset
            y_text = y_offset + text_h

            # add text to image using cv
            image_copy = image_cv.copy()
            cv2.putText(
                image_copy,
                st.session_state.title,
                (x_text, y_text),
                st.session_state.font,
                font_scale,
                st.session_state.font_color[::-1],
                st.session_state.font_thickness
            )

            #display final image
            image_final = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            st.subheader("Final Album Art with Title")
            st.image(image_final, use_column_width=True)

else:
    st.warning("⛔ Please enter both artist and album name to continue.")


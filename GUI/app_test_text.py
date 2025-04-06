import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Get the absolute path of the Scripts directory
script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Add Scripts/ to the system path
sys.path.append(script_dir)

from title_placement import place_text_using_visual_balance

# Set Streamlit page config
st.set_page_config(page_title="Text Placement Demo", layout="centered")
st.title("📐 Visual Balance Title Placement")
st.write("Upload an image and visualize the optimal text placement using the Visual Balance algorithm.")

# Ask user to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Font options (OpenCV fonts)
font_options = {
    "Simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "Complex": cv2.FONT_HERSHEY_COMPLEX,
    "Duplex": cv2.FONT_HERSHEY_DUPLEX,
    "Triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "Simplex Script": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    "Complex Script": cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    "Plain": cv2.FONT_HERSHEY_PLAIN,
}


if uploaded_file is not None:
    # Store the uploaded image in session_state
    if "original_image" not in st.session_state:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.original_image = image
    else:
        image = st.session_state.original_image

    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image_cv.shape

    st.header("Choose Font, Text Size, and Color")

    text_size_option = st.selectbox("Select Text Size", ["Big", "Medium", "Small"])
    if text_size_option == "Big":
        text_size = int(image_width * 0.75), 60
    elif text_size_option == "Medium":
        text_size = int(image_width * 0.50), 60
    else:
        text_size = int(image_width * 0.25), 60

    selected_font = st.selectbox("Select Font", list(font_options.keys()))
    font = font_options[selected_font]

    color_hex = st.color_picker("Pick Text Color", "#FFFFFF")
    text_color = tuple(int(color_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    thickness = st.slider("Text Thickness", min_value=1, max_value=10, value=2)


    #place title once the user pushes the button
    if st.button("Place Text"):
        text = "Sample Text"
        position = place_text_using_visual_balance(image_cv, text_size=text_size, use_diagonal=True)
        x, y = position if position else (100, 100)

        st.session_state.text = text
        st.session_state.font = font
        st.session_state.text_size = text_size
        st.session_state.font_name = selected_font
        st.session_state.font_color = text_color
        st.session_state.x = x
        st.session_state.y = y
        st.session_state.placed = True

    # once we calculate position place text on image and add adjustment sliders
    if st.session_state.get("placed", False):
        st.header("Adjust Text Position")
        x_offset = st.slider("X Offset", min_value=0, max_value=image_width, value=st.session_state.x, step=1)
        y_offset = st.slider("Y Offset", min_value=0, max_value=image_height, value=st.session_state.y, step=1)

        # calculate font scale based on user selected size
        def calculate_font_scale(text, max_width, max_height, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
            font_scale = 1.0
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            if text_width < max_width:
                while text_width < max_width and text_height < max_height:
                    font_scale += 0.1
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            else:
                while text_width > max_width:
                    font_scale -= 0.1
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            return font_scale

        #adjust font based on user selected size
        font_scale = calculate_font_scale(
            st.session_state.text,
            st.session_state.text_size[0],
            st.session_state.text_size[1],
            font=st.session_state.font
        )
        (text_width, text_height), _ = cv2.getTextSize(st.session_state.text, st.session_state.font, font_scale, thickness)

        #adjust possition to have correct text placement
        x_text = x_offset
        y_text = y_offset + text_height

        #add text
        image_with_text = cv2.cvtColor(np.array(st.session_state.original_image), cv2.COLOR_RGB2BGR)
        cv2.putText(image_with_text, st.session_state.text, (x_text, y_text), st.session_state.font, font_scale, st.session_state.font_color[::-1], thickness)
        image_with_text = cv2.cvtColor(image_with_text, cv2.COLOR_BGR2RGB)

        #show image
        st.subheader("Final Image with Text Placement")
        st.image(image_with_text, caption="Text Placement", use_column_width=True)
        st.success(f"Text position: ({x_offset}, {y_offset}), Font Scale: {font_scale:.2f}")

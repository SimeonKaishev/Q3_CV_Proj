import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("CV Project GUI")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale example
    if st.button("Convert to Grayscale"):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(gray, caption="Grayscale Image", use_column_width=True, channels="GRAY")

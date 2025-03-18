import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title of the app
st.title("Album Art Generator")

# Step 1: Ask for artist name, album name, and optional inputs
artist_name = st.text_input("Artist Name")
album_name = st.text_input("Album Name")
lyrics = st.text_area("Lyrics (Optional)")
instructions = st.text_area("Instructions (Optional)")

# Store images in session state
if 'image' not in st.session_state:
    st.session_state['image'] = None

if artist_name and album_name:
    # Step 2: Allow user to either generate an image or upload one
    image_choice = st.radio(
        "What would you like to do next?",
        ("Generate an image", "Upload an image")
    )

    if image_choice == "Generate an image":

        #--------------TODO-----------------
        # Here is where we would call the functions that will use the fine-tuned sd model to generate several possible options for the user 
        #to choose from.
        #For now i just paste some text to make the image

        # Logic for generating an image
        if st.button("Generate Image"):
            st.write(f"Generating an image for artist: {artist_name}, album: {album_name}")

            # Placeholder for the image generation process
            # For now, creating a blank image with the artist and album name on it
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

            # Save the generated image to session state
            st.session_state['image'] = generated_img
            st.image(st.session_state['image'], caption="Generated Image", use_container_width=True)

    elif image_choice == "Upload an image":
        # Step 3: Allow user to upload an image
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
            uploaded_image = np.array(uploaded_image)

            # Save the uploaded image to session state
            st.session_state['image'] = uploaded_image
            st.image(st.session_state['image'], caption="Uploaded Image", use_container_width=True)

    # Step 3: Option to stylize the image
    if st.session_state['image'] is not None:
        stylize_choice = st.radio(
            "Do you want to stylize the image?",
            ("Yes", "No")
        )

        if stylize_choice == "Yes":
        
            #--------------TODO-----------------
            # Here is where the mood input would come in and we would once again generate a few versions of the image for them to choose 
            # based off of the mood. Probably 8 stylised and the original (or 2 or 5 idk depends how fucky we get with the style generatuion)
            #for now just some chat gpt blur


            # Placeholder for the stylization process
            # Here, you could add actual code for image stylization (like applying a filter)
            st.write("Stylizing the image...")

            # Example: Apply a simple blur filter to simulate stylization
            stylized_img = cv2.GaussianBlur(st.session_state['image'], (15, 15), 0)

            st.image(stylized_img, caption="Stylized Image", use_container_width=True)
        else:
            st.write("No stylization applied. Here's the original image.")
            st.image(st.session_state['image'], caption="Original Image", use_container_width=True)

    # ------------------TODO--------------------
    #Add the text inserting bit here
else:
    st.warning("Please enter both artist and album name to proceed.")
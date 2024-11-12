import streamlit as st
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image
from io import BytesIO

def renoise(img):
    img_cv = np.array(img)
    img_med = cv.medianBlur(img_cv,5)
    #img_out = reblur(img_med)
    return img_med

def reblur(img):
    smoothed_image = cv.GaussianBlur(img, (5, 5), 0)

    laplacian1 = cv.Laplacian(smoothed_image, cv.CV_64F)

    laplacian2 = cv.Laplacian(laplacian1, cv.CV_64F)

    img_out = cv.convertScaleAbs(laplacian2)
    return img_out

def increaselight(img):
    img_cv = np.array(img)
    img_out = cv.convertScaleAbs(img_cv, alpha=1.2, beta=30)
    return img_out

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def convert_img_to_bytes(img_array):
    # Convert the numpy array to a PIL Image
    img_pil = Image.fromarray(img_array)
    
    # Save the image to a BytesIO object
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)  # Rewind the BytesIO object to the beginning
    return img_bytes

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("my_model.h5")  # Assuming the model is saved in 'SavedModel' format
    return model

def show_upload():
    load_css
    st.header('Detail of Model')
    st.markdown("""
    - Model 1: Explain Model 1 here
    - Model 2: Explain Model 2 here
    - Model 3: Explain Model 3 here
    """)
    st.markdown('<div><h2>Convert image</h2></div>', unsafe_allow_html=True)
    st.markdown('<div><h5>You can reduce noise ,blur and increase light here </h5></div>', unsafe_allow_html=True)

    option = st.selectbox(
    "How would you like to be contacted?",
    ("Reduce Blur", "Reduce Noise", "Increase Light"),
    index=None,
    placeholder="Select contact method...",
    )
    st.write("You selected:", option)

    uploaded_file_convert = st.file_uploader("Choose an image file to convert image", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded_file_convert is not None:
        img_con = None
        # Open the uploaded image and ensure it is in RGB format
        image = Image.open(uploaded_file_convert).convert("RGB")
        image_np = np.array(image)
        if option == 'Reduce Blur':
            img_con = reblur(image_np)
        elif option == 'Reduce Noise':
            img_con = renoise(image_np)
        elif option == 'Increase Light':
            img_con = increaselight(image_np)
        else: img_con = image_np

        st.image(img_con, caption="Uploaded Image", use_container_width=True)
        if not np.array_equal(img_con, image_np):
            img_bytes = convert_img_to_bytes(img_con)
            st.download_button(label='Download Image',data= img_bytes,file_name="trans_image.png",mime="image/png")
        else: st.write("No transformation was applied.")
    

    
    model = load_model()
    st.markdown('<div><h2>Select Model to predict image</h2></div>', unsafe_allow_html=True)
    uploaded_file_model = st.file_uploader("Choose an image file to predict image", type=["png", "jpg", "jpeg"], key="uploadermodel")

    if uploaded_file_model is not None:
        # Open the uploaded image and ensure it is in RGB format
        image = Image.open(uploaded_file_model).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        img_array = np.array(image.resize((224, 224))) 
        img_array = img_array / 255.0  # Normalize the image data
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        
        # Display the result
        st.write("Prediction:", prediction)


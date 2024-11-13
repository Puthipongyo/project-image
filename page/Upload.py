import streamlit as st
import tensorflow as tf
import numpy as np
import cv2 as cv
import base64
from tensorflow import keras
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import preprocess_input


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
def load_model(option_model):
    model = None
    if option_model == 'Model 1':
        model = tf.keras.models.load_model("modelnew.h5")  # Assuming the model is saved in 'SavedModel' format
    elif option_model == 'Model 2':
        model = tf.keras.models.load_model("modelhighaccuracy.h5")
    return model

def predict(image_file, model):
    if image_file is not None:
        if model is None:
            st.error("No model selected or model loading failed.")
            return
        # Open the uploaded image and ensure it is in RGB format
        image = Image.open(image_file).convert("RGB")
        
        #st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        img_array = np.array(image.resize((224, 224))) 
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)
        if predicted_class[0] == 0:
            st.write("This picture is FAKE")
        else:
            st.write("This picture is REAL")
        
        # Display the result
        st.write("Fake Probility : ",np.round(prediction[0][0] * 100, 2),'%')
        st.write("Real Probility : ",np.round(prediction[0][1] * 100, 2),'%')
        
def show_upload():
    load_css()
    
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
    

    option_model = st.selectbox(
    "How would you like to use model?",
    ("Model 1", "Model 2"),
    index=None,
    placeholder="Select contact method...",
    )
    st.write("You selected:", option_model)
    
    model = load_model(option_model)
    st.markdown('<div><h2>Select Model to predict image</h2></div>', unsafe_allow_html=True)
    uploaded_file_model = st.file_uploader("Choose an image file to predict image", type=["png", "jpg", "jpeg"], key="uploadermodel")

    if uploaded_file_model is not None:
        buffered = BytesIO()
        image = Image.open(uploaded_file_model).convert("RGB")
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

    # Embed the base64 image in the HTML
        st.markdown(
            f"""
            <div class="upload-image">
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" style="width:100%;">
            </div>
            """,unsafe_allow_html=True
        )
    
    # Run prediction if the model is loaded
        predict(uploaded_file_model, model)
    else:
        st.write("Please upload an image file.")


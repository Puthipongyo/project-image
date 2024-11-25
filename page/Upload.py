import streamlit as st
import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.resnet50 import preprocess_input

def renoise(img):
    img_cv = np.array(img)
    img_med = cv.medianBlur(img_cv,5)
    img_out = reblur(img_med)
    return img_out

def reblur(img):
    kernel_sharpening = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    output_sharpened = cv.filter2D(img, -1, kernel_sharpening)
    return output_sharpened

def increaselight(img):
    img_cv = np.array(img)
    img_out = cv.convertScaleAbs(img_cv, alpha=1.2, beta=30)
    return img_out

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def convert_img_to_bytes(img_array):
    img_pil = Image.fromarray(img_array)
    img_bytes = BytesIO()
    img_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

@st.cache_resource
def load_model(option_model):
    if option_model is None:
            st.error("Please Select Model.")
            return
    if option_model == 'Model 1':
        model = tf.keras.models.load_model("Model1.h5")
    elif option_model == 'Model 2':
        model = tf.keras.models.load_model("Model2.h5")
    elif option_model == 'Model 3':
        model = tf.keras.models.load_model("Model3.h5")
    return model

def predict(image_file, model):
    if image_file is not None:
        if model is None:
            st.error("No model selected or model loading failed.")
            return
        
        image = Image.open(image_file).convert("RGB")
        
        img_array = np.array(image.resize((224, 224))) 
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array)
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=-1)
        if predicted_class[0] == 0:
            st.write("This picture is FAKE")
        else:
            st.write("This picture is REAL")
        
        st.write("Fake Probility : ",np.round(prediction[0][0] * 100, 2),'%')
        st.write("Real Probility : ",np.round(prediction[0][1] * 100, 2),'%')  
        
def show_upload():
    load_css()
    
    st.header('Detail of Model')
    st.markdown(""" 
    - Model 1 (Basic CNN Model) : Can classify images within the dataset but poor results when use with images outside the dataset.
    - Model 2 (ResNet50) : Can classify both images within the dataset and those outside the dataset.
    - Model 3 (ResNet50) with Fine-Tuning : Can classify both images within the dataset and those outside the dataset and usually better result that Model 2
    """)
    st.markdown('<div><h2>Convert image</h2></div>', unsafe_allow_html=True)
    st.markdown('<div><h5>You can reduce noise ,blur and increase light here </h5></div>', unsafe_allow_html=True)

    option = st.selectbox(
        "How would you like to be contacted?",
        ("Reduce Blur", "Reduce Noise", "Increase Light"),
        index=None,
        placeholder="Select contact method..."
    )
    st.write("You selected:", option)
    uploaded_file = st.file_uploader("Choose an image file to convert image", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded_file is not None:
        img_con = None
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        if option == 'Reduce Blur':
            img_con = reblur(image_np)
        elif option == 'Reduce Noise':
            img_con = renoise(image_np)
        elif option == 'Increase Light':
            img_con = increaselight(image_np)
        else: 
            img_con = image_np

        st.image(img_con, caption="Uploaded Image", use_container_width=True)

        if not np.array_equal(img_con, image_np):
            img_bytes = convert_img_to_bytes(img_con)
            st.download_button(label='Download Image', data=img_bytes, file_name="trans_image.png", mime="image/png")
        else:
            st.write("No transformation was applied.")

        st.markdown('<div><h2>Choose Model</h2></div>', unsafe_allow_html=True)
        
        
        option_model = st.selectbox(
            "How would you like to use model?",
            ("Model 1", "Model 2","Model 3"),
            index=None,
            placeholder="Select contact method..."
        )
        st.write("You selected:", option_model)
        
        model = load_model(option_model)

        if model and uploaded_file is not None:
            image_model = convert_img_to_bytes(img_con)
            predict(image_model, model)

    else:
        st.write("Please upload an image file to proceed.")




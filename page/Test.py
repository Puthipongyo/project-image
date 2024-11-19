import os
import random
import streamlit as st
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from page.Upload import predict
from page.Upload import load_model
from PIL import Image
from io import BytesIO

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
def predict_visual(image_file, model):
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

def print_image_visual(img):
    image = Image.open(img).convert("RGB")
        
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
            f"""
            <div class="visual-image">
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" style="width:100%;">
            </div>
            """,
            unsafe_allow_html=True
        )

def get_random_image(image_folder):
    all_images = os.listdir(image_folder)
    image_files = [f for f in all_images if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    return os.path.join(image_folder, random.choice(image_files)) if image_files else None


def show_test():
    load_css()

    fake_folder = "src/fake"
    real_folder = "src/real"

    st.markdown('<h1 class="center-header">AI Image Example</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        image_path = get_random_image(fake_folder)
        if image_path:
            print_image_visual(image_path)
        else:
            st.write("No images found in fake folder.")
    with col3:
        st.markdown('<h4>Model 1</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 1'))
        st.markdown('<h4>Model 2</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        st.markdown('<h4>Model 3</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        
    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        image_path = get_random_image(real_folder)
        if image_path:
            print_image_visual(image_path)
        else:
            st.write("No images found in fake folder.")
    with col6:
        st.markdown('<h4>Model 1</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 1'))
        st.markdown('<h4>Model 2</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        st.markdown('<h4>Model 3</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))


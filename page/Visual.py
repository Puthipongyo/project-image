import os
import random
import streamlit as st
import base64
from page.Upload import predict
from page.Upload import load_model
from PIL import Image
from io import BytesIO

def predict_visual(img):
    image = Image.open(img).convert("RGB")
        
        # Convert image to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
        # Embed the base64 image in the HTML
    st.markdown(
            f"""
            <div class="visual-image">
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" style="width:100%;">
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('<h3> Model 1 </h3>', unsafe_allow_html=True)
    predict(img,load_model('Model 1'))
    st.markdown('<h3> Model 2 </h3>', unsafe_allow_html=True)
    predict(img,load_model('Model 2'))

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_random_image(image_folder):
    # List all files in the directory
    all_images = os.listdir(image_folder)
    # Filter only image files (optional, based on common image extensions)
    image_files = [f for f in all_images if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    # Randomly select an image and return the full file path
    return os.path.join(image_folder, random.choice(image_files)) if image_files else None

def show_visual():
    # Load custom CSS and model
    load_css()
    # Set paths to "fake" and "real" folders
    fake_folder = "src/fake"
    real_folder = "src/real"

    # Display header for AI (Fake) images
    st.markdown('<h1 class="center-header">AI Image Example</h1>', unsafe_allow_html=True)

    # Create columns for displaying random images from the "fake" folder
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict_visual(image_path)
    with col2:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict_visual(image_path)
    with col3:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict_visual(image_path)
    with col4:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict_visual(image_path)
    
    # Display header for Real images
    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    # Create columns for displaying random images from the "real" folder
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        image_path = get_random_image(real_folder)
        if image_path:
            predict_visual(image_path)
    with col6:
        image_path = get_random_image(real_folder)
        if image_path:
            predict_visual(image_path)
    with col7:
        image_path = get_random_image(real_folder)
        if image_path:
            predict_visual(image_path)
    with col8:
        image_path = get_random_image(real_folder)
        if image_path:
            predict_visual(image_path)

    # Display comparison graph
    st.write('Graph to compare each model (bar chart)')

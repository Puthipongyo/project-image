import os
import random
import streamlit as st
from page.Upload import predict
from page.Upload import load_model


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
    option_model = st.selectbox(
    "How would you like to use model?",
    ("Model 1", "Model 2"),
    index=None,
    placeholder="Select contact method...",
    )
    st.write("You selected:", option_model)
    
    model = load_model(option_model)

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
            predict(image_path, model)
    with col2:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict(image_path, model)
    with col3:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict(image_path, model)
    with col4:
        image_path = get_random_image(fake_folder)
        if image_path:
            predict(image_path, model)
    
    # Display header for Real images
    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    # Create columns for displaying random images from the "real" folder
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        image_path = get_random_image(real_folder)
        if image_path:
            predict(image_path, model)
    with col6:
        image_path = get_random_image(real_folder)
        if image_path:
            predict(image_path, model)
    with col7:
        image_path = get_random_image(real_folder)
        if image_path:
            predict(image_path, model)
    with col8:
        image_path = get_random_image(real_folder)
        if image_path:
            predict(image_path, model)

    # Display comparison graph
    st.write('Graph to compare each model (bar chart)')

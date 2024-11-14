import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from tensorflow import keras
from PIL import Image
from io import BytesIO
from page.Upload import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import random

def predict_forgame(image_file, model):
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
            st.write("Model answer --> This picture is FAKE")
        else:
            st.write("Model answer --> This picture is REAL")

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_random_image(image_folder):
    all_images = os.listdir(image_folder)
    image_files = [f for f in all_images if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    return os.path.join(image_folder, random.choice(image_files)) if image_files else None

def show_game():
    load_css()

    # Define the folders containing real and fake images
    fake_folder = "src/fake"
    real_folder = "src/real"
    list_folder = [fake_folder, real_folder]
    
    # Clear the session state when coming to this page to reload the image
    if 'selected_folder' in st.session_state:
        del st.session_state.selected_folder
    if 'image_path' in st.session_state:
        del st.session_state.image_path
    
    # Now, select the folder and image again
    selected_folder = np.random.choice(list_folder)
    st.session_state.selected_folder = selected_folder
    
    image_path = get_random_image(selected_folder)
    st.session_state.image_path = image_path
    
    # Show the folder the image was selected from
    st.write(f"Selected folder: {selected_folder}")  # Print the selected folder
    
    # Display the image
    buffered = BytesIO()
    image = Image.open(image_path).convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
            f"""
            <div class="upload-image">
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" style="width:100%;">
            </div>
            """, unsafe_allow_html=True
        )
    
    # Provide a unique key for each selectbox to avoid duplicate ID conflict
    if 'user_answer' not in st.session_state:
        st.session_state.user_answer = None  # Initialize the user_answer in session state if not present

    # The user selects an answer in the selectbox
    user_answer = st.selectbox(
        "Is this image real or fake?", 
        ["Real", "Fake"], 
        index=None if st.session_state.user_answer is None else ["Real", "Fake"].index(st.session_state.user_answer),  # Reset the selection based on session state
        help="Select 'Real' or 'Fake' to answer"
    )

    # Update the session state when user selects an answer
    if user_answer != st.session_state.user_answer:
        st.session_state.user_answer = user_answer  # Store the answer in session state
    
    # Error handling: Ensure user selects an answer
    if user_answer is None:
        st.error("Please select an answer")
    else:
        # Check if the answer matches the selected folder
        if selected_folder == fake_folder:
            st.markdown('<h4>This picture is fake </h4>', unsafe_allow_html=True)
        elif selected_folder == real_folder:
            st.markdown('<h4>This picture is real </h4>', unsafe_allow_html=True)
        
        st.write(f"Your answer: {user_answer}")
        
        # Load the model and make a prediction
        model = load_model('Model 1')
        predict_forgame(image_path, model)
        
        # Add selectbox under the predict_forgame function
    # Button to clear the session state and reload the page
    if st.button("New image"):
        # Clear the session state for the selected folder, image path, and user answer
        if 'selected_folder' in st.session_state:
            del st.session_state.selected_folder
        if 'image_path' in st.session_state:
            del st.session_state.image_path
        if 'user_answer' in st.session_state:
            del st.session_state.user_answer
        st.rerun()  # Rerun the app to reload the image and reset the game
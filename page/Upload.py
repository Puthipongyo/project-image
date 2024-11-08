import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_model.h5")

def show_upload():
    st.header('Detail of Model')
    st.markdown("""
    - Model 1: Explain Model 1 here
    - Model 2: Explain Model 2 here
    - Model 3: Explain Model 3 here
    """)
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"], key="uploader")

    if uploaded_file is not None:
        # Open the uploaded image and ensure it is in RGB format
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess the image
        img_array = np.array(image.resize((224, 224))) 
        img_array = img_array / 255.0  # Normalize the image data
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        
        # Display the result
        st.write("Prediction:", prediction)


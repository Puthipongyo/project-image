import streamlit as st
from PIL import Image

def show_upload():
    st.header('Detail of model')
    st.markdown("""
    - Model 1 : (Explain)
    - Model 2 : (Explain)
    - Model 3 : (Explain)
    """)
    
    # Add a file uploader widget to accept only image files
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        image = Image.open(uploaded_file)
        st.write(f"Image size: {image.size}")
        st.write(f"Image mode: {image.mode}")

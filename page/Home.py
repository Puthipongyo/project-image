import streamlit as st

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the CSS
load_css()

def show_home():
    # Header of home page
    st.markdown('<h1 class="center-header">AI-Generated vs. Real Image Classification</h1>', unsafe_allow_html=True)

    st.image("src/bird.png", caption="AI Image 1", width=650 )
    # content
    st.write("How to use this app")
    

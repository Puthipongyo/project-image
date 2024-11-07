import streamlit as st

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Apply the CSS
load_css()

def show_visual():
    # Header AI image
    st.markdown('<h1 class="center-header">AI Image Example</h1>', unsafe_allow_html=True)
     
    #  Create column
    col1, col2, col3, col4 = st.columns(4)

    # Place the images in each of the columns
    with col1:
        st.image("src/ai.png", caption="AI Image 1", width=150)
    with col2:
        st.image("src/ai.png", caption="AI Image 2", width=150)
    with col3:
        st.image("src/ai.png", caption="AI Image 3", width=150)
    with col4:
        st.image("src/ai.png", caption="AI Image 4", width=150)
    
    # Header Real image
    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    #  Create column
    col5, col6, col7, col8 = st.columns(4)

    # Place the images in each of the columns
    with col5:
        st.image("src/ai.png", caption="AI Image 1", width=150)
    with col6:
        st.image("src/ai.png", caption="AI Image 2", width=150)
    with col7:
        st.image("src/ai.png", caption="AI Image 3", width=150)
    with col8:
        st.image("src/ai.png", caption="AI Image 4", width=150)
    st.write('graph to compare each model (bar chart)')

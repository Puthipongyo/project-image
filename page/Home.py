import streamlit as st

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def show_home():
    load_css()
    st.markdown('<h1 class="center-header"> RE-AL or FA-KE </h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="center-header"> Image Classification</h3>', unsafe_allow_html=True)

    st.image("src/Home.jpeg", width=650)

    st.markdown('''<h6 class ="about-container-home"> <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;These days, artificial intelligence (AI) is becoming more and more integrated 
                into our daily lives, offering enormous advantages in many different domains. It does, however, have certain disadvantages. This is especially noticeable in the art industry, where problems with copyright infringement and ownership have surfaced. This project has been developed to address this issue by utilising machine learning and image processing techniques to differentiate between AI-generated graphics and artwork created by humans. Training dataset is from Kaggle. </h6>'''
                , unsafe_allow_html=True)

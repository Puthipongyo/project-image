import streamlit as st
import base64
from PIL import Image
from io import BytesIO
def convert_image(img):
    image = Image.open(img).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def show_about():
    load_css()
    st.markdown('<div style="margin:50px;"><h1 class="center-header">Team Members</h1></div>', unsafe_allow_html=True)
    img_nana = convert_image('src/Nana.jpg')
    img_pleum = convert_image('src/Pleum.jpg')
    img_phat = convert_image('src/Phat.jpg')
    img_ter = convert_image('src/Ter.jpg')
    st.markdown(
    f"""
    <div class="about-container">
        <div>
            <div class="circle-image">
                <img src="data:image/jpeg;base64,{img_nana}" alt="Nipatsa Chainiwattana">
            </div>
            <div class="name-text">Nipatsa Chainiwattana</div>
        </div>
        <div>
            <div class="circle-image">
                <img src="data:image/jpeg;base64,{img_ter}" alt="Puthipong Yomabut">
            </div>
            <div class="name-text">Puthipong Yomabut</div>
        </div>
        <div>
            <div class="circle-image">
                <img src="data:image/jpeg;base64,{img_pleum}" alt="Patiharn Kamenkit">
            </div>
            <div class="name-text">Patiharn Kamenkit</div>
        </div>
        <div>
            <div class="circle-image">
                <img src="data:image/jpeg;base64,{img_phat}" alt="Phattaradanai Sornsawang">
            </div>
            <div class="name-text">Phattaradanai Sornsawang</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True)

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

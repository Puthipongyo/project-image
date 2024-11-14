import streamlit as st


def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_about():
    load_css()
    st.markdown('<div style="margin:50px;"><h1 class="center-header">Team Members</h1></div>', unsafe_allow_html=True)

    st.markdown(
    """
    <div class="about-container">
        <div>
            <div class="circle-image">
                <img src="https://staticg.sportskeeda.com/editor/2023/02/394a3-16769313907566-1920.jpg" alt="Nipatsa Chainiwattana">
            </div>
            <div class="name-text">Nipatsa Chainiwattana</div>
        </div>
        <div>
            <div class="circle-image">
                <img src="https://staticg.sportskeeda.com/editor/2023/02/394a3-16769313907566-1920.jpg" alt="Puthipong Yomabut">
            </div>
            <div class="name-text">Puthipong Yomabut</div>
        </div>
        <div>
            <div class="circle-image">
                <img src="https://staticg.sportskeeda.com/editor/2023/02/394a3-16769313907566-1920.jpg" alt="Patiharn Kamenkit">
            </div>
            <div class="name-text">Patiharn Kamenkit</div>
        </div>
        <div>
            <div class="circle-image">
                <img src="https://staticg.sportskeeda.com/editor/2023/02/394a3-16769313907566-1920.jpg" alt="Phattaradanai Sornsawang">
            </div>
            <div class="name-text">Phattaradanai Sornsawang</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


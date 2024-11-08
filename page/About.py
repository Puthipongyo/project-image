import streamlit as st

# Load the CSS file
def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_about():
    load_css()
    st.markdown('<div style="margin:50px;"><h1 class="center-header">About Us</h1></div>', unsafe_allow_html=True)
    
    # List of names
    Name = ["Nipatsa Chainiwattana", "Puthipong Yomabut", "Patiharn Kamenkit", "Phattaradanai Sornsawang"]
    
    # Create four columns for the four names
    cols = st.columns(4)  # Create four columns once
    
    # Loop through each name and display it in the corresponding column
    for i, name in enumerate(Name):
        with cols[i]:  # Access each column dynamically using square brackets
            st.markdown(
                '<div class="circle-image" style = margin: 50px;><img src="https://staticg.sportskeeda.com/editor/2023/02/394a3-16769313907566-1920.jpg"></div>',
                unsafe_allow_html=True
            )
            st.markdown(f"""
                <div style= padding-top: 10px; margin: 50px;">
                    <h3>{name}</h3>
                </div>
            """, unsafe_allow_html=True)


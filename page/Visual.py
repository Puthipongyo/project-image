import os
import random
import streamlit as st
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
from page.Upload import predict
from page.Upload import load_model
from PIL import Image
from io import BytesIO

def predict_visual(img):
    image = Image.open(img).convert("RGB")
        
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
            f"""
            <div class="visual-image">
                <img src="data:image/jpeg;base64,{img_str}" alt="Uploaded Image" style="width:100%;">
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('<h3> Model 1 </h3>', unsafe_allow_html=True)
    predict(img, load_model('Model 1'))
    st.markdown('<h3> Model 2 </h3>', unsafe_allow_html=True)
    predict(img, load_model('Model 2'))

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def get_random_image(image_folder):
    all_images = os.listdir(image_folder)
    image_files = [f for f in all_images if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    return os.path.join(image_folder, random.choice(image_files)) if image_files else None

def visualize_graph():
    labels = ['AI painting pictures', 'REAL painting pictures']
    sizes = [10330, 8288]
    colors = ['#ff3b31', '#66b3ff']  # Custom colors for each slice

    # Create a pie chart with custom colors and add a centered title
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(colors=colors),  # Set the custom colors
    )])

    # Add a centered title to the figure layout
    fig.update_layout(
        title={
            'text': "Segment of AI and Real Image",
            'y': 0.9,
            'x': 0.4,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        title_font_size=24
    )

    st.plotly_chart(fig)

    nations = ["Training", "Testing", "Validation"]
    gold = [39, 38, 37]
    silver = [41, 32, 28]

    # Create a bar chart using Plotly
    fig = go.Figure()

    # Add bars for each medal type
    fig.add_trace(go.Bar(x=nations, y=gold, name="Gold", text=gold, textposition='outside'))
    fig.add_trace(go.Bar(x=nations, y=silver, name="Silver", text=silver, textposition='outside'))

    # Customize the layout
    fig.update_layout(
    title={
        'text': "Painting Pictures Count",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 24}
    },
    barmode='stack',  # Stacked bars
    xaxis={'title': {'text': "Picture"}},
    yaxis={'title': {'text': "Category"}}
)
    # Show the figure in Streamlit
    st.plotly_chart(fig)

def show_visual():
    load_css()

    fake_folder = "src/fake"
    real_folder = "src/real"

    st.markdown('<h1 class="center-header">AI Image Example</h1>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    for col in [col1, col2, col3, col4]:
        with col:
            image_path = get_random_image(fake_folder)
            if image_path:
                predict_visual(image_path)
            else:
                st.write("No images found in fake folder.")

    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    col5, col6, col7, col8 = st.columns(4)
    for col in [col5, col6, col7, col8]:
        with col:
            image_path = get_random_image(real_folder)
            if image_path:
                predict_visual(image_path)
            else:
                st.write("No images found in real folder.")

    visualize_graph()

import os
import random
import streamlit as st
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from page.Upload import predict
from page.Upload import load_model
from PIL import Image
from io import BytesIO

def load_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
def predict_visual(image_file, model):
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
            st.write("This picture is FAKE")
        else:
            st.write("This picture is REAL")
        
        st.write("Fake Probility : ",np.round(prediction[0][0] * 100, 2),'%')
        st.write("Real Probility : ",np.round(prediction[0][1] * 100, 2),'%')

def print_image_visual(img):
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

def get_random_image(image_folder):
    all_images = os.listdir(image_folder)
    image_files = [f for f in all_images if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    return os.path.join(image_folder, random.choice(image_files)) if image_files else None


def show_visual():
    load_css()

    fake_folder = "src/fake"
    real_folder = "src/real"

    st.markdown('<h1 class="center-header">AI Image Example</h1>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        image_path = get_random_image(fake_folder)
        if image_path:
            print_image_visual(image_path)
        else:
            st.write("No images found in fake folder.")
    with col3:
        st.markdown('<h4>Model 1</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 1'))
        st.markdown('<h4>Model 2</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        st.markdown('<h4>Model 3</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        
    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    col4, col5, col6 = st.columns(3)
    with col4:
        image_path = get_random_image(real_folder)
        if image_path:
            print_image_visual(image_path)
        else:
            st.write("No images found in fake folder.")
    with col6:
        st.markdown('<h4>Model 1</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 1'))
        st.markdown('<h4>Model 2</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        st.markdown('<h4>Model 3</h4>', unsafe_allow_html=True)
        predict(image_path, load_model('Model 2'))
        
    visualize_graph()


def visualize_graph():
    labels = ['AI painting pictures', 'REAL painting pictures']
    sizes = [10330, 8288]
    colors = ['#E44C51', '#4C7AE4']  

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

    options = ["Training", "Testing", "Validation"]
    AI = [18292, 4160, 4153]
    real = [7847, 1783, 1781]
    colors_AI = '#E44C51'  # สีแดงสำหรับ AI
    colors_real = '#4C7AE4'  # สีน้ำเงินสำหรับ Real

    fig = go.Figure()

    ## bar chart
    fig.add_trace(go.Bar(
        x=options, 
        y=real, 
        name="Real", 
        text=real, 
        textposition='outside',
        marker=dict(color=colors_real)  
    ))

    fig.add_trace(go.Bar(
        x=options, 
        y=AI, 
        name="AI", 
        text=AI, 
        textposition='outside',
        marker=dict(color=colors_AI)  
    ))

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
        yaxis={'title': {'text': "Picture"}},
        xaxis={'title': {'text': "Category"}}
    )
    # Show the figure in Streamlit
    st.plotly_chart(fig)

    # ResNet model data
    epoch = [1, 2, 3, 4, 5, 6]
    loss = [0.3031206727, 0.2488057315, 0.239877671, 0.2289397269, 0.2233790457, 0.2149953097]
    val_loss = [0.249250859, 0.233581394, 0.222523436, 0.2166575938, 0.2160844207, 0.2322244644]
    accuracy = [0.8784957528, 0.9012586474, 0.9060790539, 0.910420596, 0.9115115404, 0.915566802]
    val_accuracy = [0.8985360861, 0.9099781513, 0.9153625965, 0.9123338461, 0.9109877348, 0.9050984383]

    # Create Accuracy figure
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=accuracy, name='Training Accuracy',
                                    line=dict(color='#4CE4B1', width=4, dash='dash')))
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name='Validation Accuracy',
                                    line=dict(color='#4CE4B1', width=4)))
    accuracy_fig.update_layout(
        title=dict(text='Accuracy of ResNet Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Accuracy')),
    )

    # Create Loss figure
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epoch, y=loss, name='Training Loss',
                                line=dict(color='firebrick', width=4, dash='dash')))
    loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, name='Validation Loss',
                                line=dict(color='firebrick', width=4)))
    loss_fig.update_layout(
        title=dict(text='Loss of ResNet Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Loss')),
    )

    col1, col2 = st.columns(2)

    # Display plots in respective columns
    with col1:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    with col2:
        st.plotly_chart(loss_fig, use_container_width=True)


    # mobilenet model
    epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    loss = [0.3252242804, 0.2867548168, 0.2747557163, 0.2661762238, 0.2616060078, 0.2545778751, 
            0.2479001731, 0.2477663159, 0.2445758879]
    val_loss = [0.2964640856, 0.2790726721, 0.2592705786, 0.2726657987, 0.2556690276, 0.2529106438,
                0.261025108, 0.2563704848, 0.2529461086]
    accuracy = [0.86793679, 0.8863766789, 0.8914266229, 0.8937602639, 0.8967443109, 0.8996136189,
                0.901090533, 0.9027889371, 0.901056423]
    val_accuracy = [0.8855796456, 0.8827191591, 0.8961803913, 0.8923102617, 0.89483428, 0.8971899748,
                    0.8956755996, 0.8938246965, 0.8960121274]

    # Create Accuracy figure
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=accuracy, name='Training Accuracy',
                                    line=dict(color='#E49D4C', width=4, dash='dash')))
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name='Validation Accuracy',
                                    line=dict(color='#E49D4C', width=4)))
    accuracy_fig.update_layout(
        title=dict(text='Accuracy of Mobilenet Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Accuracy')),
    )

    # Create Loss figure
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epoch, y=loss, name='Training Loss',
                                line=dict(color='firebrick', width=4, dash='dash')))
    loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, name='Validation Loss',
                                line=dict(color='firebrick', width=4)))
    loss_fig.update_layout(
        title=dict(text='Loss of Mobilenet Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Loss')),
    )

    col1, col2 = st.columns(2)

    # Display plots in respective columns
    with col1:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    with col2:
        st.plotly_chart(loss_fig, use_container_width=True)


     # last model
    epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    loss = [0.3252242804, 0.2867548168, 0.2747557163, 0.2661762238, 0.2616060078, 0.2545778751, 
            0.2479001731, 0.2477663159, 0.2445758879]
    val_loss = [0.2964640856, 0.2790726721, 0.2592705786, 0.2726657987, 0.2556690276, 0.2529106438,
                0.261025108, 0.2563704848, 0.2529461086]
    accuracy = [0.86793679, 0.8863766789, 0.8914266229, 0.8937602639, 0.8967443109, 0.8996136189,
                0.901090533, 0.9027889371, 0.901056423]
    val_accuracy = [0.8855796456, 0.8827191591, 0.8961803913, 0.8923102617, 0.89483428, 0.8971899748,
                    0.8956755996, 0.8938246965, 0.8960121274]

    # Create Accuracy figure
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=accuracy, name='Training Accuracy',
                                    line=dict(color='pink', width=4, dash='dash')))
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name='Validation Accuracy',
                                    line=dict(color='pink', width=4)))
    accuracy_fig.update_layout(
        title=dict(text='Accuracy of Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Accuracy')),
    )

    # Create Loss figure
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epoch, y=loss, name='Training Loss',
                                line=dict(color='firebrick', width=4, dash='dash')))
    loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, name='Validation Loss',
                                line=dict(color='firebrick', width=4)))
    loss_fig.update_layout(
        title=dict(text='Loss of Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Loss')),
    )

    col1, col2 = st.columns(2)

    # Display plots in respective columns
    with col1:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    with col2:
        st.plotly_chart(loss_fig, use_container_width=True)
   

import os
import random
import streamlit as st
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
# from page.Upload import predict
# from page.Upload import load_model
from Upload import predict
from Upload import load_model
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

    image_path = get_random_image(fake_folder)
    print_image_visual(image_path)
    st.write("")
    col1, col2, col3 = st.columns(3)
    if image_path:
        with col1:    
            st.markdown('<h4>Model 1</h4>', unsafe_allow_html=True)
            predict(image_path, load_model('Model 1'))
        with col2:
            st.markdown('<h4>Model 2</h4>', unsafe_allow_html=True)
            predict(image_path, load_model('Model 2'))
        with col3:
            st.markdown('<h4>Model 3</h4>', unsafe_allow_html=True)
            predict(image_path, load_model('Model 3'))
    else:
        st.write("No images found in fake folder.")
        
    st.markdown('<h1 class="center-header">Real Image Example</h1>', unsafe_allow_html=True)

    image_path = get_random_image(real_folder)
    print_image_visual(image_path)
    st.write("")
    col4, col5, col6 = st.columns(3)
    if image_path:
        with col4:    
            st.markdown('<h4>Model 1</h4>', unsafe_allow_html=True)
            predict(image_path, load_model('Model 1'))
        with col5:
            st.markdown('<h4>Model 2</h4>', unsafe_allow_html=True)
            predict(image_path, load_model('Model 2'))
        with col6:
            st.markdown('<h4>Model 3</h4>', unsafe_allow_html=True)
            predict(image_path, load_model('Model 3'))
    else:
        st.write("No images found in fake folder.")
    
        
    visualize_graph()


def visualize_graph():
    labels = ['AI painting pictures', 'REAL painting pictures']
    sizes = [10330, 8288]
    colors = ['#E44C51', '#4C7AE4']  

    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        textinfo='label+percent',
        insidetextorientation='radial',
        marker=dict(colors=colors), 
    )])

    
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
    colors_AI = '#E44C51' 
    colors_real = '#4C7AE4' 

    fig = go.Figure()

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

    fig.update_layout(
        title={
            'text': "Painting Pictures Count",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        barmode='stack',  
        yaxis={'title': {'text': "Picture"}},
        xaxis={'title': {'text': "Category"}}
    )

    st.plotly_chart(fig)

    # CNN 
    epoch = [1, 2, 3, 4, 5]
    loss = [0.694229, 0.592310, 0.515308, 0.490537, 0.473285]
    accuracy = [0.552852, 0.703279, 0.776656, 0.794369, 0.802441]
    val_loss = [0.656276, 0.530266, 0.488951, 0.542080, 0.502169]
    val_accuracy = [0.644960, 0.789164, 0.794548, 0.745920, 0.782938]


    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=accuracy, name='Training Accuracy',
                                    line=dict(color='#4CE4B1', width=3, dash='dash')))
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name='Validation Accuracy',
                                    line=dict(color='#4CE4B1', width=3)))
    accuracy_fig.update_layout(
        title=dict(text='Accuracy of Simple CNN Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Accuracy')),
    )


    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epoch, y=loss, name='Training Loss',
                                line=dict(color='firebrick', width=3, dash='dash')))
    loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, name='Validation Loss',
                                line=dict(color='firebrick', width=3)))
    loss_fig.update_layout(
        title=dict(text='Loss of Simple CNN Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Loss')),
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    with col2:
        st.plotly_chart(loss_fig, use_container_width=True)



    ## Resnet
    epoch = [1, 2, 3, 4, 5]
    loss = [0.3782561421, 0.3311806023, 0.3202846348, 0.3099855781, 0.3165609241]
    accuracy = [0.8469719291, 0.8708825707, 0.8750143647, 0.8810971975, 0.8795669079]
    val_loss = [0.4360812604, 0.3923163712, 0.3274979591, 0.4336121678, 0.2789599895]
    val_accuracy = [0.8189466596, 0.833754003, 0.8623591065, 0.8244994283, 0.8921419978]



    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=accuracy, name='Training Accuracy',
                                    line=dict(color='#E49D4C', width=3, dash='dash')))
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name='Validation Accuracy',
                                    line=dict(color='#E49D4C', width=3)))
    accuracy_fig.update_layout(
        title=dict(text='Accuracy of Resnet Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Accuracy')),
    )


    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epoch, y=loss, name='Training Loss',
                                line=dict(color='firebrick', width=3, dash='dash')))
    loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, name='Validation Loss',
                                line=dict(color='firebrick', width=3)))
    loss_fig.update_layout(
        title=dict(text='Loss of Resnet Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Loss')),
    )

    col1, col2 = st.columns(2)


    with col1:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    with col2:
        st.plotly_chart(loss_fig, use_container_width=True)


    # Resnet fine tune
    epoch = [1, 2, 3, 4, 5]
    loss = [0.366931170225143, 0.30858364701271, 0.287208646535873, 0.281542807817459, 0.271793335676193]
    val_loss = [0.321894079446792, 0.274576783180236, 0.274013966321945, 0.229892373085021, 0.246545508503913]
    accuracy = [0.852251410484314, 0.878572225570678, 0.890240609645843, 0.89307165145874, 0.89376026391983]
    val_accuracy = [0.872959792613983, 0.89702171087265, 0.896348655223846, 0.910146415233612, 0.90526670217514]


    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=accuracy, name='Training Accuracy',
                                    line=dict(color='#d950ff', width=3, dash='dash')))
    accuracy_fig.add_trace(go.Scatter(x=epoch, y=val_accuracy, name='Validation Accuracy',
                                    line=dict(color='#d950ff', width=3)))
    accuracy_fig.update_layout(
        title=dict(text='Accuracy of Resnet fine tune Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Accuracy')),
    )


    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epoch, y=loss, name='Training Loss',
                                line=dict(color='firebrick', width=3, dash='dash')))
    loss_fig.add_trace(go.Scatter(x=epoch, y=val_loss, name='Validation Loss',
                                line=dict(color='firebrick', width=3)))
    loss_fig.update_layout(
        title=dict(text='Loss of Resnet fine tune Model'),
        xaxis=dict(title=dict(text='Epoch')),
        yaxis=dict(title=dict(text='Loss')),
    )

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(accuracy_fig, use_container_width=True)

    with col2:
        st.plotly_chart(loss_fig, use_container_width=True)
   


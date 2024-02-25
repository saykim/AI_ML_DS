import streamlit as st
from autogluon.multimodal import MultiModalPredictor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io
import time
import os
import glob
import base64

# Change the page configuration
st.set_page_config(
    page_title="Image Classification Demo",
    page_icon=":factory:",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'About': "# 이해를 돕기 위한 화면입니다. 테스트용도로만 사용하세요!"}
)

# Set up custom CSS for the app
custom_css = """
<style>
    body {
        background-color: #F5F5F5;
    }
    .reportview-container .markdown-text-container {
        font-family: monospace;
        color: #000000;
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .fullScreenFrame > div {
        max-width: 100%;
    }
    h1 {
        color: #4F4F4F;
        font-size: 40px;
    }
    h2 {
        color: #828282;
        font-size: 30px;
    }
    h3 {
        color: #BDBDBD;
        font-size: 20px;
    }
    h4 {
        color: #E0E0E0;
        font-size: 15px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 모델 로드
predictor = MultiModalPredictor.load("autogluon_model.ag")

st.title('Image Classification Demo \n by Sangyeon, mfg intelligence')

# Cumulative results
results = {"GOOD": 0, "NG": 0}

def get_image_html(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    encoded = base64.b64encode(image_bytes.getvalue()).decode()
    html = f'<img src="data:image/jpeg;base64,{encoded}" style="display: block; margin-left: auto; margin-right: auto;">'
    return html

# 이미지 업로드를 통한 예측
uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file).convert('RGB')
    image.thumbnail((image.width // 2, image.height // 2))  # Resize the image to 50%
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("AppleGothic.ttf", 30)
    
    # Convert the image to bytes and then to a DataFrame
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_df = pd.DataFrame({'image': [image_bytes.getvalue()]})
    
    # Get the prediction and the probabilities
    start_time = time.time()
    result = predictor.predict(image_df)
    proba = predictor.predict_proba(image_df)
    inference_time = time.time() - start_time
    
    # Draw the prediction and the inference time on the image
    label = 'GOOD' if result[0] == 0 else 'NG'
    color = 'green' if label == 'GOOD' else 'red'
    draw.text((50, 30), label, fill=color, font=font)
    draw.text((50, 60), f"Inference Time: {round(inference_time, 3)} seconds", fill=color, font=font)
    draw.rectangle([(1, 1), (image.width, image.height)], outline=color, width=10)
    
    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'Image': [1],  # This is a single image upload
        'Label': ['GOOD' if r == 0 else 'NG' for r in result],
        'GOOD %': [f"{round(p * 100, 3)}%" for p in proba[0]],
        'NG %': [f"{round(p * 100, 3)}%" for p in proba[1]],
        'Inference Time': [f"{round(inference_time, 3)} seconds"]
    })
    
    box_color = '#8DFFA9' if label == 'GOOD' else '#FF8D8D'  # green for 'GOOD', red for 'NG'
    st.markdown(f"<div style='background-color: {box_color}; padding: 10px; border-radius: 5px;'>"
                f"<h3 style='color: black; text-align: center;'>Prediction: {label}</h3>"
                f"<h4 style='color: black; text-align: center;'>Inference Time: {round(inference_time, 3)} seconds</h4>"
                f"</div>", unsafe_allow_html=True)

    # Show the image and the prediction
    st.markdown(get_image_html(image), unsafe_allow_html=True)

    st.table(result_df)

# 폴더 위치를 통한 예측
folder_path = st.text_input("Enter the folder path...")
result_placeholder = st.empty()  # Placeholder for the results
image_placeholder = st.empty()  # Placeholder for the images

if st.button('Start Inspection'):
    if os.path.isdir(folder_path):
        images = glob.glob(os.path.join(folder_path, '*.jpg'))
        for i, image_path in enumerate(images, start=1):  # Add a counter to track the image order
            st.write(f"Processing: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image.thumbnail((image.width // 2, image.height // 2))  # Resize the image to 50%
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("AppleGothic.ttf", 30)

            # Convert the image to bytes and then to a DataFrame
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            image_df = pd.DataFrame({'image': [image_bytes.getvalue()]})
            
            # Get the prediction and the probabilities
            start_time = time.time()
            result = predictor.predict(image_df)
            proba = predictor.predict_proba(image_df)
            inference_time = time.time() - start_time
            
            # Draw the prediction on the image
            label = 'GOOD' if result[0] == 0 else 'NG'
            color = 'green' if label == 'GOOD' else 'red'
            draw.text((50, 30), label, fill=color, font=font)
            draw.text((50, 60), f"Inference Time: {round(inference_time, 3)} seconds", fill=color, font=font)
            draw.rectangle([(1, 1), (image.width, image.height)], outline=color, width=10)
            
            # Create a DataFrame with the results
            result_df = pd.DataFrame({
                'Image': [i],  # Add the image order
                'Label': [label],
                'GOOD %': [f"{round(p * 100, 3)}%" for p in proba[0]],
                'NG %': [f"{round(p * 100, 3)}%" for p in proba[1]],
                'Inference Time': [f"{round(inference_time, 3)} seconds"]
            })
            
            box_color = '#8DFFA9' if label == 'GOOD' else '#FF8D8D'  # green for 'GOOD', red for 'NG'
            result_placeholder.markdown(f"<div style='background-color: {box_color}; padding: 10px; border-radius: 5px;'>"
                        f"<h3 style='color: black; text-align: center;'>Prediction: {label}</h3>"
                        f"<h4 style='color: black; text-align: center;'>Inference Time: {round(inference_time, 3)} seconds</h4>"
                        f"</div>", unsafe_allow_html=True)
            
            # Show the image and the prediction
            image_placeholder.markdown(get_image_html(image), unsafe_allow_html=True)
            
            # Show the results
            st.table(result_df)

            # Update the cumulative results
            results[label] += 1
            st.markdown(f"**Cumulative Results: {results}**")

            # Wait for 2 seconds before processing the next image
            time.sleep(2)
    else:
        st.write(f"The folder {folder_path} does not exist.")

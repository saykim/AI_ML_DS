import streamlit as st
from keras.models import load_model
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
import io
import time
import os
import glob

# 모델 로드
model = load_model("MobileNetV2.h5")

st.title('Image Classification Demo \n by Sangyeon, mfg intelligence')

# Cumulative results
results = {"GOOD": 0, "NG": 0}

# 이미지 업로드를 통한 예측
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.thumbnail((image.width // 2, image.height // 2))  # Resize the image to 50%
    st.image(image, caption='Uploaded Image.')
    st.write("")
    st.write("Classifying...")
    
    # Convert the image to bytes and then to a DataFrame
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')
    image_df = pd.DataFrame({'image': [image_bytes.getvalue()]})
    
    # Get the prediction and the probabilities
    start_time = time.time()
    result = model.predict(image_df)
    proba = model.predict_proba(image_df)
    inference_time = time.time() - start_time
    
    # Create a DataFrame with the results
    result_df = pd.DataFrame({
        'Label': ['GOOD' if r == 0 else 'NG' for r in result],
        'GOOD %': [f"{round(p * 100, 3)}%" for p in proba[0]],
        'NG %': [f"{round(p * 100, 3)}%" for p in proba[1]],
        'Inference Time': [f"{round(inference_time, 3)} seconds"]
    })
    
    st.table(result_df)

# 폴더 위치를 통한 예측
folder_path = st.text_input("Enter the folder path...")
if st.button('Start Inspection'):
    if os.path.isdir(folder_path):
        images = glob.glob(os.path.join(folder_path, '*.jpg'))
        image_placeholder = st.empty()  # Placeholder for the images
        result_placeholder = st.empty()  # Placeholder for the results
        for image_path in images:
            st.write(f"Processing: {image_path}")
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))  # Resize the image to (224, 224)
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("AppleGothic.ttf", 30)

            # Convert the image to numpy array and preprocess it
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)  # Add a new dimension at the 0th position
            img_array = preprocess_input(img_array)
                        
            # Get the prediction and the probabilities
            start_time = time.time()
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)  # Add a new dimension at the 0th position
            img_array = preprocess_input(img_array)
            result = model.predict(img_array)
            proba = result  # This line is changed
            inference_time = time.time() - start_time

            
            # Draw the prediction on the image
            label = 'GOOD' if result[0] == 0 else 'NG'
            color = 'green' if label == 'GOOD' else 'red'
            draw.text((100, 100), label, fill=color, font=font)
            draw.rectangle([(1, 1), (image.width, image.height)], outline=color, width=10)
            
            # Show the image and the prediction
            image_placeholder.image(image, caption='Predicted Image.')

            # Create a DataFrame with the results
            result_df = pd.DataFrame({
                'Label': [label],
                'GOOD %': [f"{round(p * 100, 3)}%" for p in proba[0]],
                'NG %': [f"{round(p * 100, 3)}%" for p in proba[1]],
                'Inference Time': [f"{round(inference_time, 3)} seconds"]
            })
            
            # Show the results
            result_placeholder.table(result_df)

            # Update the cumulative results
            results[label] += 1
            st.write(f"Cumulative Results: {results}")

            # Wait for 2 seconds before processing the next image
            time.sleep(2)
    else:
        st.write(f"The folder {folder_path} does not exist.")

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# Load the TensorFlow Lite model
model = tf.lite.Interpreter("model.tflite")
model.allocate_tensors()

# Get input and output tensors
input_details = model.get_input_details()
output_details = model.get_output_details()

# Preprocess the input image
def preprocess_image(image):
    image = image.resize((input_details[0]['shape'][1], input_details[0]['shape'][2]))
    image = np.array(image)
    image = image / 255.0
    image = image.astype(np.float32)
    return image

# Run inference on the input image
def classify_image(model, image):
    input_data = np.array(preprocess_image(image), dtype=np.uint8)
    input_data = np.expand_dims(input_data, axis=0)
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

# Create a Streamlit app
st.set_page_config(page_title="Image Classification", page_icon=":camera:", layout="wide")
st.title("Image Classification with TensorFlow Lite")

# Get the input image
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, use_column_width=True)
    
# Classify the image
    output = classify_image(model, image)
    label = "Class: " + str(np.argmax(output))
    st.write(label)
    st.write(output)
    

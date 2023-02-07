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
    class_labels = {'Class 1':"1", 'Class 2':"2",'Class 3': "3",'Class 4': "4",'Class 5': "5",'Class 6': "6",'Class 7': "7",'Class 8': "8",'Class 9': "9",
                    'Class 10':"A",'Class 11': "B",'Class 12': "C",'Class 13': "D",'Class 14': "E",'Class 15': "F",'Class 16': "G",'Class 17': "H",'Class 18': "I",'Class 19':"J",
                    'Class 20': "K",'Class 21': "L",'Class 22': "M",'Class 23': "N",'Class 24': "O",'Class 25': "P",'Class 26': "Q",'Class 27': "R",'Class 28': "S",'Class 29': "T",
                    'Class 30':"U",'Class 31': "V",'Class 32': "W",'Class 33': "X",'Class 34': "Y",'Class 35': "Z"}
    label = np.argmax(output)
    class_label = class_labels[label]
    print(class_labels.value())


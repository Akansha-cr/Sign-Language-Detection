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
    class_labels = {1:"1", 2:"2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10:"A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
                    19:"J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T", 30:"U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"}
    class_index = (np.argmax(output)
    class_label = class_labels[label]
    label = "Class: " + str(np.argmax(output))
    st.write(class_label)
    st.write(label)
    

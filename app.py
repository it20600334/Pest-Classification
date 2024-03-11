# import flast module
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# instance of flask application
app = Flask(__name__)

# Load the model
# model_path = '/path/to/your/model.h5'  # Replace with your model path
# model = load_model(model_path)
# class_labels = ['CCI', 'NI', 'WF']  # Replace with your class labels

def preprocess_image(img):
    # Preprocess the image (resize, preprocess_input)
    img = cv2.resize(img, (224, 224))  # Adjust target_size if needed
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array



# home route that returns below text when root url is accessed
@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__': 
    app.run()
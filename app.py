from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tf.keras.models import load_model
from tf.keras.preprocessing import image
from tf.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import base64
import os

app = Flask(__name__)
# Load your trained model
# model = load_model('E:\\SLIIT\\Year 4\\Research\\Application\\Pest-Classification\\MobileNetV2OP.h5')
model = load_model('MobileNetV2OP.h5')

# Map class indices to class labels
class_labels = {
    0: 'CCI',
    1: 'NI',
    2: 'WF'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the POST request contains the 'image' field
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    # Get the base64-encoded image data from the request
    image_data = request.json['image'].split(',')[1]

    # Decode the base64-encoded image data
    img_bytes = base64.b64decode(image_data)
    print(img_bytes)

    # Save the decoded image to a temporary file
    temp_img_path = 'temp_image.jpg'
    with open(temp_img_path, 'wb') as img_file:
        img_file.write(img_bytes)

    # Perform image preprocessing
    img = image.load_img(temp_img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array = preprocess_input(img_array) 

    # Use the loaded model for prediction
    result = model.predict(img_array)

    # Map the predicted class index to its label
    predicted_class_index = result.argmax(axis=1)
    predicted_class_label = class_labels[predicted_class_index[0]]

    # Get the confidence score (probability) for the predicted class
    confidence_score = result[0][predicted_class_index[0]]

    # Remove the temporary image file
    os.remove(temp_img_path)

    # Return the prediction result and confidence score
    return jsonify({'result': predicted_class_label, 'confidence': float(confidence_score)})


if __name__ == '__main__': 
    app.run()
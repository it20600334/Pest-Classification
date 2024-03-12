from flask import Flask, request, jsonify, requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Flask app initialization
app = Flask(__name__)

# Model path (replace with your actual model path)
model_path = '/path/to/your/MobileNetV2OP.h5'

# Load the model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))  # Adjust target_size as per your model's input shape
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image using MobileNetV2 preprocessing function
    return img_array

# Prediction function
def make_prediction(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    class_labels = ['CCI', 'NI', 'WF']  # Replace with your actual class labels
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Access uploaded image data
    img_file = request.files['image']
    img_path = '/path/to/save/images/' + img_file.filename
    img_file.save(img_path)

    # Perform prediction
    prediction_result = make_prediction(img_path)

    # Send prediction result to the frontend via JSON response
    return requests.post('http://your_frontend_url/update_prediction', json={'prediction': prediction_result})  # Replace with actual frontend URL

if __name__ == '__main__':
    app.run(debug=True)

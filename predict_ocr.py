import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('crnn_model.h5')

# Function to preprocess the image for prediction
def preprocess_for_prediction(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 32))  # Resize to match model input
    image = image.astype('float32') / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to decode the predicted output
def decode_predictions(predictions):
    # Implement a way to decode the predicted output to readable text
    pass  # You need to implement this based on your model output

# Main prediction code
image_path = 'images/your_image.jpeg'  # Change this to your test image
preprocessed_image = preprocess_for_prediction(image_path)
predictions = model.predict(preprocessed_image)

# Decode and print predictions
decoded_text = decode_predictions(predictions)
print(f"Predicted Text: {decoded_text}")

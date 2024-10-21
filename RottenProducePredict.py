import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_path = 'Models/rotten_fruit_model.keras'
model = load_model(model_path)

# Load class indices from JSON
with open('Models/rotten_class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Invert the dictionary to map indices to class labels
class_labels = {v: k for k, v in class_indices.items()}

# Function to predict the fruit type
def predict_fruit(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))  # Resize to match model's expected input
    img_array = img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to [0, 1]

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the class index of the highest probability
    predicted_label = class_labels[predicted_class_index]  # Get the predicted class label

    return predicted_label

def get_predict(img_path):
    # Example usage
    predicted_fruit = predict_fruit(img_path)
    print(f"Predicted Fruit: Rotten {predicted_fruit}")

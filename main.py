import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import PredictProduct
import EasyOCR_DatesAndRS
import PredictFreshness
import FreshProducePredict
import RottenProducePredict

# Load the trained model
model = tf.keras.models.load_model('Models/fruit_vs_product_model.h5')

# Define class labels
class_labels = ['PRODUCTS', 'fresh', 'rotten']


# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize to the model's input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array


# Function to predict the class of an image
def predict_image(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)  # Get predictions
    predicted_class = class_labels[np.argmax(predictions)]  # Get the class with the highest probability
    return predicted_class, predictions

def call_Funcs(predicted_class, img_path):
    if predicted_class == 'fresh' or predicted_class == 'rotten':
        # Freshness.get_prediction()
        freshness = PredictFreshness.get_freshness(img_path)
        print(f"Product is {freshness}")
        if freshness == 'Fresh':
            FreshProducePredict.get_predict(img_path)

        elif freshness == 'Rotten':
            RottenProducePredict.get_predict(img_path)

    elif(predicted_class == "PRODUCTS"):
        print(f'Predicted Class: {predicted_class}')
        PredictProduct.get_prediction(img_path)
        EasyOCR_DatesAndRS.get_Text_exp_manf_mrp(img_path)


if __name__ == "__main__":

    for file in os.listdir('testData'):
        img_path = os.path.join('testData', file)
        print(f'\n{img_path}')
        # Ask user for image file path
        img_path = img_path

        # Check if the file exists
        try:
            predicted_class, predictions = predict_image(img_path)
            call_Funcs(predicted_class, img_path)
        except Exception as e:
            print(f"Error: {e}")


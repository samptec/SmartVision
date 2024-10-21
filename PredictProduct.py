import os

import easyocr
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
import re

# Initialize EasyOCR reader for text extraction
reader = easyocr.Reader(['en'])  # Add more languages if necessary

# Load the trained product classification model
model = tf.keras.models.load_model('Models/product_classification_model.keras')

# List of product names (replace with actual product names from the dataset)
product_names = os.listdir('data/train')  # Replace '...' with actual product names


# Function to preprocess the input image for the classification model
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, target_size)  # Resize to match the model input size (224x224)
    img_normalized = img_resized / 255.0  # Normalize pixel values (0-1)
    img_expanded = np.expand_dims(img_normalized, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return img_expanded

def increase_contrast(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the channels back
    limg = cv2.merge((cl, a, b))
    contrast_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return contrast_image

# Function to extract text from the image using EasyOCR
def extract_text_from_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    image = cv2.filter2D(image, -1, kernel)

    # Perform OCR using EasyOCR
    result = reader.readtext(image)

    # Extract the detected text
    detected_text = [text for (bbox, text, prob) in result]

    cv2.waitKey(0)
    return detected_text


# Function to split product names by both spaces and underscores and match extracted text
def find_best_match_by_words(extracted_text, product_names):
    # Flatten the extracted text list into a set of words (lowercased)
    extracted_words = set(word.lower() for text in extracted_text for word in text.split())

    match_counts = Counter()

    # Compare each product name with the extracted words
    for product_name in product_names:
        # Split the product name by both underscores and spaces using regex
        product_words = set(re.split(r'[ _]', product_name.lower()))  # Split by space and underscore

        # Count how many words from the product name match the extracted words
        common_words = extracted_words.intersection(product_words)
        match_counts[product_name] = len(common_words)

    # Find the product name with the highest number of matching words
    best_match = match_counts.most_common(1)

    if best_match:
        return best_match[0]  # Return the product name and the number of matches
    else:
        return None, 0


# Function to predict the product using the image classifier
def predict_product(image_path):
    # Preprocess the image for the model
    preprocessed_image = preprocess_image(image_path)

    # Predict the class probabilities
    predictions = model.predict(preprocessed_image)

    # Get the index of the class with the highest predicted probability
    predicted_class_index = np.argmax(predictions)

    # Get the product name corresponding to the predicted class
    predicted_product_name = product_names[predicted_class_index]

    # Get the confidence score of the prediction
    confidence_score = np.max(predictions)

    return predicted_product_name, confidence_score


# Combine OCR and image classification to predict the exact product
def combined_prediction(image_path):
    # Step 1: Extract text from the image using OCR
    extracted_text = extract_text_from_image(image_path)

    # CHANGED PART: If no text is detected, skip text matching and proceed with image classification
    if extracted_text:
        # Find the product name with the most word matches, splitting by underscores and spaces
        best_match, match_count = find_best_match_by_words(extracted_text, product_names)
    else:
        # No text detected, so skip text-based matching
        best_match, match_count = None, 0

    # CHANGED PART: If no text-based match is found or text count is 0, fall back to image classification
    if match_count > 0 and best_match:
        print(f"Best text-based match: {best_match} (Matches: {match_count})")
        final_product_name = best_match
    else:
        # Use image classification if no strong text match or no text found
        product_name, image_confidence = predict_product(image_path)
        print(f"Predicted Product (Image): {product_name}, Confidence: {image_confidence:.2f}")
        final_product_name = product_name

    return final_product_name

def get_prediction(img_path):
    # Test the combined prediction with a sample image
    image_path = img_path  # Path to the image you want to predict
    final_product = combined_prediction(image_path)

    print(f"Final Predicted Product: {final_product}")
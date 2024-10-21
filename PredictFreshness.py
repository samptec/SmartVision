import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('Models/freshness_detection_model.keras')

# Function to preprocess an image for testing
def preprocess_image(img_path, target_size=(224, 224)):  # Adjust target size as needed
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
    img_array /= 255.0  # Normalize the image (if the model expects this)
    return img_array


# Function to evaluate model on a single image
def evaluate_model_on_image(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)

    # Assuming the model predicts a freshness score (e.g., a value between 0 and 1)
    freshness_score = prediction[0][0]  # Adjust based on model output shape
    print(f"Predicted Freshness Score: {100-freshness_score*100}")

    # Threshold for classification (assuming binary classification of freshness)
    if freshness_score < 0.5:
        return "Fresh"
    else:
        return "Rotten"

def get_freshness(img_path):
    # Test the model on a sample image
    freshness = evaluate_model_on_image(model, img_path)
    return freshness

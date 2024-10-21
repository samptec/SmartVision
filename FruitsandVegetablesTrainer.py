import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Set dataset path
dataset_dir = "Fruits_&_Vegetables"

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values (0-1)
    rotation_range=30,  # Rotate images for augmentation
    width_shift_range=0.2,  # Horizontal shifting
    height_shift_range=0.2,  # Vertical shifting
    shear_range=0.2,  # Shearing
    zoom_range=0.2,  # Zoom in/out
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest',  # Fill missing pixels after transformation
    validation_split=0.2  # Reserve 20% of images for validation
)

# Load training data
train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),  # Resize all images to 150x150 pixels
    batch_size=32,
    class_mode='categorical',  # Use categorical for multi-class classification
    subset='training',  # Use training subset
    shuffle=True
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use validation subset
)

# Get fruit class labels (e.g., fresh_apple, rotten_banana)
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}
print(class_labels)

# Model architecture (Convolutional Neural Network)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(len(class_indices), activation='softmax')  # Output layer for multi-class classification
])

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Save the model
model.save('Models/freshness_detection_model.h5')


# Function to predict freshness and fruit name
def predict_freshness(image_path):
    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Get freshness status and fruit name
    prediction_label = class_labels[predicted_class]
    print(f"The predicted result is: {prediction_label.replace('_', ' ')}")


# Test on a sample image
# Replace with your image path
predict_freshness('Fruits_&_Vegetables/fresh/beans cluster/5ab7324b21f92_Cluster-Beans.png')
predict_freshness('Fruits_&_Vegetables/rotten/bottle gourd/bottle-gourd-lagenaria-siceraria-isolated-black-background-indian-lauki-dudhi-240060160.jpg')

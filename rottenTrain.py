import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Directories for fresh and rotten fruit images
rotten_dir = 'Fruits_&_Vegetables/rotten'

# Define image size and batch size
image_size = (128, 128)
batch_size = 32


# Preprocessing for rotten fruits
rotten_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into 80% train, 20% validation
)

# Load rotten fruit images
rotten_train_gen = rotten_datagen.flow_from_directory(
    rotten_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Multiclass classification for different rotten fruits
    subset='training'
)

rotten_val_gen = rotten_datagen.flow_from_directory(
    rotten_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

import json

# Save class indices to a JSON file
with open('Models/rotten_class_indices.json', 'w') as f:
    json.dump(rotten_train_gen.class_indices, f)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# Define a function that returns a CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(rotten_train_gen.num_classes, activation='softmax')  # Softmax for multiclass classification
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the model for rotten fruits
rotten_model = create_model()

history_rotten = rotten_model.fit(
    rotten_train_gen,
    steps_per_epoch=rotten_train_gen.samples // batch_size,
    validation_data=rotten_val_gen,
    validation_steps=rotten_val_gen.samples // batch_size,
    epochs=30,  # Start with up to 30 epochs, it may stop early
)

# Save the rotten fruit model
rotten_model.save('Models/rotten_fruit_model.keras')

rotten_loss, rotten_acc = rotten_model.evaluate(rotten_val_gen)
print(f"Rotten Fruit Model Accuracy: {rotten_acc * 100:.2f}%")


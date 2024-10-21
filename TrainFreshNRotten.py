import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Directories for fresh and rotten fruit images
fresh_dir = 'Fruits_&_Vegetables/fresh'

# Define image size and batch size
image_size = (128, 128)
batch_size = 32

# Preprocessing for fresh fruits
fresh_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into 80% train, 20% validation
)



# Load fresh fruit images
fresh_train_gen = fresh_datagen.flow_from_directory(
    fresh_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Assuming multiclass classification for different fruits
    subset='training'
)

fresh_val_gen = fresh_datagen.flow_from_directory(
    fresh_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

import json

# Save class indices to a JSON file
with open('Models/fresh_class_indices.json', 'w') as f:
    json.dump(fresh_train_gen.class_indices, f)


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
        Dense(fresh_train_gen.num_classes, activation='softmax')  # Softmax for multiclass classification
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Create the model for fresh fruits
fresh_model = create_model()

# Train the fresh fruit model
history_fresh = fresh_model.fit(
    fresh_train_gen,
    steps_per_epoch=fresh_train_gen.samples // batch_size,
    validation_data=fresh_val_gen,
    validation_steps=fresh_val_gen.samples // batch_size,
    epochs=30,  # Start with up to 30 epochs
)

# Save the fresh fruit model
fresh_model.save('Models/fresh_fruit_model.keras')


fresh_loss, fresh_acc = fresh_model.evaluate(fresh_val_gen)
print(f"Fresh Fruit Model Accuracy: {fresh_acc * 100:.2f}%")

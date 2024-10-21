from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models

# Define data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255.0)  # Normalizing training data
val_datagen = ImageDataGenerator(rescale=1.0/255.0)    # Normalizing validation data

# Flow training images in batches from the training directory
train_generator = train_datagen.flow_from_directory(
    'classified',  # Your main dataset directory
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # For categorical labels
)

# Debugging: Check the class indices
print("Class indices:", train_generator.class_indices)

# Flow validation images in batches from the validation directory
val_generator = val_datagen.flow_from_directory(
    'classified',  # Your main dataset directory
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # For categorical labels
)

# Debugging: Check the class indices for validation
print("Validation Class indices:", val_generator.class_indices)

# Build the CNN model
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
    layers.Dense(3, activation='softmax')  # Update to 3 for three classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the model using the generator
try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // 32,  # Adjust for batch size
        epochs=10,
        validation_data=val_generator,
        validation_steps=val_generator.samples // 32  # Adjust for validation set size
    )
except Exception as e:
    print("Error during model training:", str(e))

# Save the trained model for future use
model.save('Models/fruit_vs_product_model.h5')

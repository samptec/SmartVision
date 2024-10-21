import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define ImageDataGenerator with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # normalize pixel values between 0 and 1
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and augment the images from your 'data' folder
train_generator = datagen.flow_from_directory(
    'data/',  # root directory of your data
    target_size=(224, 224),  # resizing to match pre-trained model input
    batch_size=32,
    class_mode='categorical'  # or 'binary' if it's a two-class problem
)

# Output the augmented images to check
for images, labels in train_generator:
    print(images.shape)  # (32, 224, 224, 3)
    break  # stop after one batch

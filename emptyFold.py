import os
import shutil

def delete_empty_product_folders(directory):
    # Define a set of image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory, topdown=False):
        # Skip deleting the top-level folders (train, valid)
        if root == directory:
            continue

        # Check if the folder contains image files
        contains_image = any(file.lower().endswith(ext) for file in files for ext in image_extensions)

        # If the folder doesn't contain any images or is empty, delete it
        if not contains_image and not files:
            print(f"Deleting empty folder: {root}")
            shutil.rmtree(root)
        elif not contains_image and files:
            print(f"Folder {root} does not contain images, deleting it.")
            shutil.rmtree(root)

# Specify the paths to train and valid folders
train_folder = 'data/train'
valid_folder = 'data/validation'

# Delete empty product folders in train and valid directories
delete_empty_product_folders(train_folder)
delete_empty_product_folders(valid_folder)

print("Empty product folders deleted.")

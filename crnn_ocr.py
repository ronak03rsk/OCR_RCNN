# import pandas as pd
# import cv2
# import numpy as np
# import os

# # Load labels from the Excel sheet
# excel_file_path = 'output_dataset.xlsx'  # Use relative path or adjust as needed
# data = pd.read_excel(excel_file_path)

# # Directory where images are stored
# image_dir = 'Augmented Dataset'  # Folder where images are saved

# # Preprocess function to resize and normalize the image
# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Image not found: {image_path}")
#     image = cv2.resize(image, (128, 32))  # Resize image to fixed size
#     image = image.astype('float32') / 255.0  # Normalize to [0, 1]
#     return image

# # Load images and corresponding labels
# images = []
# labels = []

# for idx, row in data.iterrows():
#     # Extract image name and label from the correct columns
#     image_path = os.path.join(image_dir, row['image_name'])  # Replace 'image_name' with the actual column name
#     label = row['label']  # Replace 'label' with the actual column name for labels
    
#     print(f"Processing image: {image_path}")  # Debugging output to verify image paths
#     image = preprocess_image(image_path)
    
#     images.append(image)
#     labels.append(label)

# # Convert list to numpy array for further processing
# images = np.array(images)
# labels = np.array(labels)

# print("Images and labels loaded and preprocessed.")

import pandas as pd
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GRU, Dense, Dropout, TimeDistributed, Flatten, Bidirectional
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

# Load labels from the Excel sheet
excel_file_path = 'output_dataset.xlsx'  # Use relative path or adjust as needed
data = pd.read_excel(excel_file_path)

# Ensure labels are strings
data['label'] = data['label'].astype(str)  # Convert labels to strings

# Directory where images are stored
image_dir = 'Augmented Dataset'  # Folder where images are saved

# Preprocess function to resize and normalize the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    image = cv2.resize(image, (128, 32))  # Resize image to fixed size
    image = image.astype('float32') / 255.0  # Normalize to [0, 1]
    return image

# Load images and corresponding labels
images = []
labels = []

for idx, row in data.iterrows():
    # Extract image name and label from the correct columns
    image_path = os.path.join(image_dir, row['image_name'])  # Replace 'image_name' with the actual column name
    label = str(row['label'])  # Ensure the label is a string
    
    # Uncomment the following line for debugging
    # print(f"Processing image: {image_path}")  
    image = preprocess_image(image_path)
    
    images.append(image)
    labels.append(label)

# Convert list to numpy array for further processing
images = np.array(images)
labels = np.array(labels)

print("Images and labels loaded and preprocessed.")

# Create a character index mapping
unique_chars = sorted(set(''.join(labels)))  # Get unique characters from labels
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}

# Prepare labels for sequence training
def encode_labels(labels, char_to_index):
    encoded_labels = []
    for label in labels:
        encoded = [char_to_index[char] for char in label]  # Convert each character to its corresponding index
        encoded_labels.append(encoded)
    return pad_sequences(encoded_labels, padding='post')  # Pad sequences to the same length

# Encode the labels
encoded_labels = encode_labels(labels, char_to_index)

# Assuming the number of classes corresponds to the unique characters
num_classes = len(unique_chars)

# Reshape images for model input
# (number of samples, height, width, channels)
images = images.reshape(images.shape[0], 32, 128, 3)  # Adjust based on your image size

# Define the CRNN model
model = Sequential()

# CNN layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 3)))  # Adjust the input shape
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))

# Reshape output for RNN layers
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(GRU(128, return_sequences=True)))  # RNN layer
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax activation

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, np.expand_dims(encoded_labels, -1), epochs=50, batch_size=32)  # Adjust the number of epochs and batch size


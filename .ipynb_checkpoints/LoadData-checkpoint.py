#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('C:/k_CBIS-DDSM/csv.csv')

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    base_path = 'C:/k_CBIS-DDSM'
    full_path = os.path.join(base_path, img_path)
    
    img = cv2.imread(full_path)
    
    if img is None:
        raise ValueError(f"Image not found or unable to load: {full_path}")
    
    # Convert to RGB if the image has 4 channels (e.g., RGBA)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    return img

images = []
labels = []
for index, row in data.iterrows():
    img_path = row['jpg_fullMammo_img_path']  
    breast_density = row['breast density'] - 1  
    
    try:
        img = load_and_preprocess_image(img_path)
        images.append(img)
        labels.append(breast_density)
    except ValueError as e:
        print(e)  # Optionally log the error or handle it differently

X = np.array(images)
y = to_categorical(np.array(labels), num_classes=4)

# Verify the shapes
print(f'X shape: {X.shape}, y shape: {y.shape}')

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the data
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)


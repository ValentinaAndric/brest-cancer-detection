#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('breast_density_model.h5')

# Load the test dataset
test_data = pd.read_csv('C:/k_CBIS-DDSM/csv_test.csv')
img_paths = test_data['jpg_fullMammo_img_path']
true_labels = test_data['breast density'] - 1  # Subtract 1 to make labels 0-indexed

# Preprocess a single image (consistent with training preprocessing)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {img_path}")
    
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input shape
    return img

# Store predictions and true labels
predictions = []
true_labels_list = []

# Loop through the test images and make predictions
for img_file in img_paths:
    img_path = os.path.join('C:/k_CBIS-DDSM', img_file)
    img = preprocess_image(img_path)
    
    # Predict using the model
    pred = model.predict(img)
    pred_label = np.argmax(pred)
    
    # Append the predicted and true labels
    predictions.append(pred_label)
    true_labels_list.append(true_labels[img_paths.tolist().index(img_file)])

# Convert lists to numpy arrays
predictions = np.array(predictions)
true_labels_list = np.array(true_labels_list)

# Compute the confusion matrix
cm = confusion_matrix(true_labels_list, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B', 'C', 'D'], yticklabels=['A', 'B', 'C', 'D'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print the classification report
print("Classification Report:")
print(classification_report(true_labels_list, predictions, target_names=['A', 'B', 'C', 'D']))


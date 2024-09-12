import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('breast_density_model.h5')

test_data = pd.read_csv('C:/k_CBIS-DDSM/csv_test.csv')
img_paths = test_data['jpg_fullMammo_img_path']
true_labels = test_data['breast density'] - 1  

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found or unable to load: {img_path}")
    
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

predictions = []
true_labels_list = []

for img_file in img_paths:
    img_path = os.path.join('C:/k_CBIS-DDSM', img_file)
    img = preprocess_image(img_path)

    pred = model.predict(img)
    pred_label = np.argmax(pred)
    
    predictions.append(pred_label)
    true_labels_list.append(true_labels[img_paths.tolist().index(img_file)])

predictions = np.array(predictions)
true_labels_list = np.array(true_labels_list)

cm = confusion_matrix(true_labels_list, predictions)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B', 'C', 'D'], yticklabels=['A', 'B', 'C', 'D'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

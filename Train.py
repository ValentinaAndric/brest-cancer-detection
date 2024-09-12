import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_train = np.load('/content/X_train.npy')
X_val = np.load('/content/X_val.npy')
y_train = np.load('/content/y_train.npy')
y_val = np.load('/content/y_val.npy')


datagen = ImageDataGenerator(
    rotation_range=40,               
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,              
    brightness_range=[0.8, 1.2],    
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

datagen.fit(X_train)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = BatchNormalization()(x)   
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False


class_weights = compute_class_weight(class_weight='balanced', 
                                     classes=np.unique(np.argmax(y_train, axis=1)), 
                                     y=np.argmax(y_train, axis=1))
class_weights_dict = dict(enumerate(class_weights))


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9
)

model.compile(optimizer=Adam(learning_rate=lr_schedule), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  
              metrics=['accuracy'])


checkpoint = ModelCheckpoint('/content/best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', restore_best_weights=True)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)  


history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_val, y_val),
                    epochs=30,
                    class_weight=class_weights_dict,
                    callbacks=[checkpoint, early_stopping, reduce_lr])

for layer in base_model.layers[-50:]:
    layer.trainable = True


model.compile(optimizer=Adam(learning_rate=1e-5), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),  
              metrics=['accuracy'])


model.fit(datagen.flow(X_train, y_train, batch_size=32),
          validation_data=(X_val, y_val),
          epochs=10,
          class_weight=class_weights_dict,
          callbacks=[checkpoint, early_stopping, reduce_lr])


model.save('/content/breast_density_model.h5', save_format='h5')


y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred)

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=['A', 'B', 'C', 'D'], yticklabels=['A', 'B', 'C', 'D'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Normalized Confusion Matrix')
plt.show()
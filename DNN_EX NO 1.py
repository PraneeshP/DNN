import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train.shape
y_train.shape
scaler=StandardScaler()
x_train_flat=x_train.reshape(x_train.shape[0],-1)
x_test_flat=x_test.reshape(x_test.shape[0],-1)
x_train_scaled=scaler.fit_transform(x_train_flat)
x_test_scaled=scaler.transform(x_test_flat)
plt.imshow(x_train[0], cmap='gray')
plt.show()
model=tf.keras.Sequential([
    keras.layers.Dense(128,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10,activation="softmax")
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(x_train_flat,y_train,epochs=15,validation_split=0.2)
loss,accuracy=model.evaluate(x_test_flat,y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"Test Loss: {loss:.2f}")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model Accuracy')
plt.ylabel("Accuracy")
plt.xlabel('Epoch')
plt.legend(['train','validation'],loc='upper left')
plt.show()
import tensorflow as tf
from tensorflow.keras import layers
model = tf.keras.Sequential([layers.Dense(64, activation='relu', input_shape=(784,)),] ) 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
import numpy as np
train_data = np.random.random((1000, 784))  # 1000 samples, 784 features each
train_labels = np.random.randint(10, size=(1000,))  # 1000 labels, 10 classes
model.fit(train_data, train_labels, epochs=10, batch_size=32)

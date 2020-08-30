import platform
import math as m
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras as keras

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#np.set_printoptions(linewidth=200)
#plt.imshow(training_images[100])
#print(training_labels[0])
#print(training_images[0])
training_images  = training_images / 255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(512, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
#model.predict(test_images)
#print(c[1])
#print(test_labels[1])
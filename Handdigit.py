import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

# Decide whether to load an existing model or train a new one.
train_new_model = True

if train_new_model:
    # dividing and adding samples to the MNIST data set
    mnist = tf.keras.datasets.mnist
    (Traning_x_Variable, Traning_y_Variable), (x_testing_Var, y_testing_var) = mnist.load_data()

    # Normalizing the data for making length = 1
    Traning_x_Variable = tf.keras.utils.normalize(Traning_x_Variable, axis=1)
    x_testing_Var = tf.keras.utils.normalize(x_testing_Var, axis=1)

    # Creating neural network model
    # Adding one flattened input layer for the pixels, two dense hidden layers and one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # model compilation and improvement
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # educating the model(Model is being trained)
    model.fit(Traning_x_Variable, Traning_y_Variable, epochs=3)

    # assessing the model(Trained model is being accessed)
    val_loss, val_acc = model.evaluate(x_testing_Var, y_testing_var)
    print(val_loss)
    print(val_acc)

    model.save('handwritten_digits.model')                     #Saving the model
else:
    model = tf.keras.models.load_model('handwritten_digits.model')        #Loading the model

# Use them to load custom images and predict them
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Error reading image! Proceeding with next image...")
        image_number += 1
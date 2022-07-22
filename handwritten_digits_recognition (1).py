import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


print("Welcome to the Handwritten Digits Recognition program")

# Decide whether to load an existing model or train a new one, we 
new_model = True

if new_model:
    # dividing and adding samples to the MNIST data set
    mnist = tf.keras.datasets.mnist
    (X_Training_set, Y_Training_set), (X_test_Set, Y_test_Set) = mnist.load_data()

    # Normalizing the data for making length = 1
    X_Training_set = tf.keras.utils.normalize(X_Training_set, axis=1)
    X_test_Set = tf.keras.utils.normalize(X_test_Set, axis=1)

    # Creating neural network model
    # Adding one flattened input layer for the pixels, two dense hidden layers and one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # model compilation and improvement
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # educating the model(training our model)
    model.fit(X_Training_set, Y_Training_set, epochs=3)

    # assessing the model (accessing our trained model)
    val_loss, val_acc = model.evaluate(X_test_Set, Y_test_Set)
    print(val_loss)
    print(val_acc)

    model.save('handwritten_digits.model')                     #Saving the model 
else:
    model = tf.keras.models.load_model('handwritten_digits.model')        #Loading the model

# Use them to load custom images and predict them
noofimage = 1
while os.path.isfile('handwritten-digits-recognition-master/digits/digit{}.png'.format(noofimage)):
    try:
        img = cv2.imread('handwritten-digits-recognition-master/digits/digit{}.png'.format(noofimage))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("The most probable number is {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        noofimage += 1
    except:
        print("Error with currentimage tryig with another image")
        noofimage += 1
        
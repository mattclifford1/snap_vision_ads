# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
from skimage.color import rgb2gray
from tqdm import tqdm

# %%
def modeltrain():
    # storing the dataset path
    clothing_fashion_mnist = tf.keras.datasets.fashion_mnist
    
    # loading the dataset from tensorflow
    (x_train, y_train),(x_test, y_test) = clothing_fashion_mnist.load_data()
    
    # displaying the shapes of training and testing dataset
    print('Shape of training cloth images: ',
        x_train.shape)
    
    print('Shape of training label: ',
        y_train.shape)
    
    print('Shape of test cloth images: ',
        x_test.shape)
    
    print('Shape of test labels: ',
        y_test.shape)

    # storing the class names as it is
    # not provided in the dataset
    label_class_names = ['T-shirt/top', 'Trouser',
                        'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker',
                        'Bag', 'Ankle boot']

    # %% display the first images
    plt.imshow(x_train[0]) 
    plt.colorbar()  # to display the colourbar
    plt.show()

    # %% NORMALISE
    x_train = x_train / 255.0  # normalizing the training data
    x_test = x_test / 255.0  # normalizing the testing data

    # %% Plot example labels (20x20)
    plt.figure(figsize=(15, 5))  # figure size
    i = 0
    while i < 20:
        plt.subplot(2, 10, i+1)
        
        # showing each image with colourmap as binary
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        
        # giving class labels
        plt.xlabel(label_class_names[y_train[i]])
        i = i+1
        
    plt.show()  

    # %% Building the model
    cloth_model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # %% compiling the model
    cloth_model.compile(optimizer='adam',
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True),
                        metrics=['accuracy'])
    # %% Fitting the model to the training data
    cloth_model.fit(x_train, y_train, epochs=10)


    # %% Calculating loss and accuracy score
    test_loss, test_acc = cloth_model.evaluate(x_test,
                                            y_test,
                                            verbose=2)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)
    # %%
    # using Softmax() function to convert
    # linear output logits to probability
    prediction_model = tf.keras.Sequential(
        [cloth_model, tf.keras.layers.Softmax()])
    
    # feeding the testing data to the probability
    # prediction model
    return prediction_model
   
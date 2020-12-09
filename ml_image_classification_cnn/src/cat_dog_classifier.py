import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import random
#import gc
import os
import cv2 # You could either use cv2 or keras.preprocessing.image to resize the image
#from keras.preprocessing import image
#from matplotlib.pyplot import imshow
#from keras.models import Model, Sequential
from keras.models import Sequential
#from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D, Input, BatchNormalization
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical  

path = "data/train/"
IMG_SIZE = 128
NUM_CHANNELS = 3    #3 channels: RGB

#conda activate tf

'''
    Load and preprocess image data:
    Download the train dataset from Kaggle (https://www.kaggle.com/c/dogs-vs-cats/data),
    unzip and save images in the current folder under "data".
    It may take several minutes to unzip everything, but you only need to do this once. 
'''
def getData(path):
    '''
        This function return the train data and test data.
        You will fit your model on the train data and test the performance of your model on the test data.
        There are 25000 pictures in total. Ideally, the train set should have 22500 pictures and the test set has 2500.
        The shape of Xtrain should be (22500,128,128,3) and Ytrain should be (22500,),
        where 1 indicates a dog and 0 indicates a cat.
    '''
    ##################################################
    # Input your code here
    #each image is a set of pixels --> convert all those pixels into an array
    #list of pixels
    Xtrain = []
    Xtest  = []
    #list of labels (map "dog" to 1, "cat" to 0)
    Ytrain = []
    Ytest  = []
    
    #convert an image class (dog, cat) into an integer (1, 0)
    transform = lambda img_class : int(img_class == 'dog')
    
    #preprocess training images
    for i in os.listdir(path):
        #convert category
        img_class = i.split(".")[0]
        img_class = transform(img_class)
        #load an image, convert pixels into an array 
        imgs = cv2.imread(os.path.join(path, i))
        #resize the image
        imgs = cv2.resize(imgs, dsize=(IMG_SIZE, IMG_SIZE))
        #add the image and the category to a list
        if (int(i.split(".")[1]) < 11250):
            Xtrain.append(imgs)
            Ytrain.append(img_class)
        else:
            Xtest.append(imgs)
            Ytest.append(img_class)
        
    #convert lists to numpy.ndarray and reshape
    Xtrain = np.array(Xtrain).reshape(-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)        
    Ytrain = np.array(Ytrain)
    Xtest  = np.array(Xtest).reshape(-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    Ytest  = np.array(Ytest)
    #one-hot encode labels 
    Ytrain = to_categorical(Ytrain, num_classes = 2, dtype='int32') 
    Ytest  = to_categorical(Ytest, num_classes = 2, dtype='int32') 
    
    #normalize data (divide by the max value of pixel's density = 255)
    Xtrain = Xtrain/255
    Xtest = Xtest/255
    ##################################################
    print("Xtrain shape: " + str(Xtrain.shape))
    print("Ytrain shape: " + str(Ytrain.shape))
    print("Xtest shape: " + str(Xtest.shape))
    print("Ytest shape: " + str(Ytest.shape))
    return Xtrain, Ytrain, Xtest, Ytest

def getModel(input_shape):
    '''
        Design the layers of your model. You should use convolutional layers in this assignment.
        There are no requirements for what other layers you should use.
    '''
    ##################################################
    # Input your code here
   
    #define a sequential model
    model = Sequential()
    
    #add a spatial (2d-) convolutional layer: #output filters (neurons) = 16, kernel size = (3,3)
    model.add(Conv2D(16, (3,3), activation = 'relu', input_shape = input_shape))
    model.add(BatchNormalization(axis=3))
   
    #add a max pooling layer with a size of (2,2) - to select the max kernel value
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    
    #add other convolutional and max pooling layers
    model.add(Conv2D(16, (3,3), activation = 'relu'))
    model.add(BatchNormalization(axis=3))
    
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    
    model.add(Conv2D(32, (3,3), activation = 'relu'))
    model.add(BatchNormalization(axis=3))
    
    model.add(MaxPooling2D(pool_size = (2,2), strides = 2))
    
    #add a flatten layer (flattens the input to match to the dense layer)
    model.add(Flatten())
    
    #add a fully connected layer
    model.add(Dense(512, activation ='relu'))
    
    #add a SoftMax layer with 2 output units
    model.add(Dense(2, activation = 'softmax'))

    ##################################################
    return model

def main():

    Xtrain, Ytrain, Xtest, Ytest = getData(path)
    model = getModel(Xtrain.shape[1:])
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    model.fit(x = Xtrain, y = Ytrain, epochs = 10, batch_size = 256, validation_split = 0.1, shuffle = True)

    y_prob = model.predict(Xtest)
 #   pred_labels = np.array([1 if p > 0.5 else 0 for p in y_prob])

    predict = lambda predicted_label : int(predicted_label)
    pred_labels = y_prob > 0.5
    for i in range(0, pred_labels.shape[0]):
        for j in range(0, pred_labels.shape[1]):
            predict(pred_labels[i][j])
    
    accuracy = 0.0
    
    for i in range(0, pred_labels.shape[0]):
        for j in range(0, pred_labels.shape[1]):
            if (pred_labels[i][j] == Ytest[i][j]):
                accuracy += 1
    
    accuracy = accuracy/Ytest.size
 #   accuracy = sum(pred_labels == Ytest) / len(Ytest)
    assert accuracy > 0.7
    print('Your model passed the test')

if __name__=="__main__":
    main()

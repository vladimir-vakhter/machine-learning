import numpy as np
import random
from keras import layers
from keras.layers import Input, Dense
from keras.models import Model          
from keras.models import Sequential     
import keras.backend as K
from sklearn.model_selection import train_test_split 
from keras.utils.np_utils import to_categorical   

#in Anaconda Prompt
#conda create -n tf tensorflow
#conda activate tf

"""
	Implementing a multiple layer neural network using softmax on MNIST dataset.
	MNIST dataset: Handwritten digits, has a training set of 60,000 examples, and a test set of 10,000 examples.
"""
Ztrain = np.loadtxt('mnist_train.csv', delimiter = ',')     #<class 'numpy.ndarray'>    (60000, 785)
Ztest  = np.loadtxt('mnist_test.csv', delimiter = ',')      #<class 'numpy.ndarray'>    (10000, 785)
"""
	You may want to get Xtrain, Ytrain, Xtest, Ytest here.
	Because each instance in Ytrain or Ytest is a number (e.g., from 0 to 9),
    you should transform it to a vector of length 10 and the i th value being 1, others are 0.
"""
############################
# Input your code here
#number of classes (0...9)
num_class = 10                                                              
#features (60000, 784)  # 28x28=784 pixels images
Xtrain = Ztrain[: , 1:]
#labels: one-hot encoding (binary class matrix)                                                         
Ytrain = to_categorical(Ztrain[: , 0], num_classes = num_class, dtype='int32')  
#features (10000, 784)
Xtest  = Ztest[: , 1:]                                                      
Ytest = to_categorical(Ztest[: , 0], num_classes = 10, dtype='int32')
############################

# Split arrays into random train set and validation subsets.
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(Xtrain, Ytrain, test_size = 0.1, random_state = 0)

def getModel(input_shape):
    """
    	Design your model, typlically a model with a few layers is already enough for you to get a pretty good accuracy.
    	You can add whatever layers you like, and there are also no restrictions on the activation function.
    	Input: the shape of one image
    	Output: the keras model
    """
 
    ############################
    # Input your code here
    #keras model as a linear stack of fully connected layers
    model = Sequential()
    #add layers (the width of a layer = 512 nodes)
    #the input layer should have a shape that matches the shape of the training data
    model.add(Dense(512, activation = 'sigmoid', input_shape=(input_shape[0],)))
    #the hidden layer(s)
    model.add(Dense(512, activation = 'sigmoid'))
    #the output layer should have a shape that matches the number of classes
    #softmax maps our output to a [0,1] range such that the total sum = 1
    model.add(Dense(num_class, activation = 'softmax'))
    #print a summary representation of the model (number of parameters in the model)
    #model.summary()
    ############################
    
    return model

model = getModel(Xtrain.shape[1:])
"""
	Now you get the model, then you should compile it and fit data to your model.
	Don't forget that we also have the validation set, you should also use them
    when you fit your model,and you could see how your model performs on the validation dataset.
"""
############################
# Input your code here
#train and evaluate model
#use stochastic gradient descent (sgd) optimizer
#the loss value that will be minimized by the model is cross entropy (another popular choice - mean_squared_error)
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
#iterating on the data in batches of batch_size_val samples (number of samples per gradient update)
batch_size_val = 32
#the number of iterations on a dataset
num_epochs = 80
#train the model
history = model.fit(Xtrain, Ytrain,
                    batch_size = batch_size_val,
                    epochs = num_epochs,
                    verbose = False,
                    validation_data = (Xvalid, Yvalid))
############################
preds = model.evaluate(x = Xtest, y = Ytest)
print()
print("Test Accuracy = " + str(preds[1]))

assert preds[1] > 0.97
print('Your model passed the test')
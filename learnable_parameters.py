import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import  Dense,Flatten
from keras.layers.convolutional import  *
from keras.layers.pooling import *

model=Sequential([
    Conv2D(2,kernel_size=(3,3),input_shape=(20,20,3),activation='relu' ,padding='same'),
    Conv2D(3,kernel_size=(3,3),activation='relu',padding='same'),
    Flatten(),
    Dense(2,activation='softmax'),

])

model.summary()

# model with pooling layer

pooling_model= Sequential([
    Conv2D(2,kernel_size=(3,3),input_shape=(20,20,3),activation='relu',padding='some'),
    Conv2D(3,kernel_size=(3,3),activation='relu',padding='some'),
    MaxPooling2D(pool_size=(2,2),strides=2),
    Flatten(),
    Dense(2,activation='softmax'),
])

pooling_model.summary()
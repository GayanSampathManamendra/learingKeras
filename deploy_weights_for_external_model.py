import numpy as np
from random import  randint
import  keras_preprocessing
import  keras
import  tensorflow
from pandas import Series
#from sklearn.preprocessing import MinMaxScaler


from keras import  backend as k
from keras.models import  Sequential
from keras.layers import Activation
from keras.optimizers import  Adam
from keras.metrics import categorical_crossentropy
from keras.layers import Dense

#from sklearn.metrics import  confusion_matrix
import  itertools
import  matplotlib.pyplot as plt

from keras.models import load_model

external_model=Sequential([
    Dense(16,input_shape=(1,),activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(32,activation='relu'),
    Dense(2,activation='softmax')
])

external_model.load_weights('C:/Users/USER PC/Desktop/keras/keras_weight_model.h5')
print(external_model.get_weights())
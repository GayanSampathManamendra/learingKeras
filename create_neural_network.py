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





#creating the data set

train_labels=[]
train_samples=[]

for i in range(1000):
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    random_older=randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

for i in range(50):
    random_younger=randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    random_older=randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

# printing row data

for i in train_samples:
    print(i)

train_labels=np.array(train_labels)
train_samples=np.array(train_samples)


# creating a model

model=Sequential([
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

model.summary()
model.compile(Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_samples,train_labels,validation_split=0.2,batch_size=10,epochs=20,shuffle=True,verbose=2)


# creating test set of data

test_samples=[]
test_labels=[]

for i in range(10):
    random_younger=randint(13,64)
    test_samples.append(random_younger)
   # test_labels.append(1)

    random_older=randint(65,100)
    test_samples.append(random_older)
   # test_labels.append(0)

for i in range(200):
    random_younger=randint(13,64)
    test_samples.append(random_younger)
   # test_labels.append(0)

    random_older=randint(65,100)
    test_samples.append(random_older)
  # test_labels.append(1)


test_labels=np.array(test_labels)
test_samples=np.array(test_samples)

# do the prediction for test set(get probability of prediction)

predictions=model.predict(test_samples,batch_size=10,verbose=0)
for i in predictions:
    print(i)

# get predited class

rounded_prediction=model.predict_classes(test_samples,batch_size=10,verbose=0)

for i in rounded_prediction:
    print(i)

#confution metrix

#cm=confusion_matrix(test_labels,rounded_prediction)

model.save('C:/Users/USER PC/Desktop/keras/keras_Model.h5')

# only saving the architecture of the model
# loss weighs and optimaizer

json_string=model.to_json()
print(json_string)

model.save_weights('C:/Users/USER PC/Desktop/keras/keras_weight_model.h5')

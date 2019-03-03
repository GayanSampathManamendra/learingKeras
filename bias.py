from keras.models import Sequential
from keras.layers import  Dense, Activation

model=Sequential(
    [
        Dense(4,input_shape=(1,),activation='relu',use_bias=True,bias_initializer='zeros'),
        Dense(2,activation='softmax')
    ]
)

print(model.get_weights())
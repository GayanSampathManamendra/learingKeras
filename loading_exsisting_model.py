from keras.models import load_model
from keras.models import  model_from_json

#loading exsisting model - load_model('path of the model')

new_model=load_model('C:/Users/USER PC/Desktop/keras/keras_model.h5')

#new_model.summary()

print(new_model.get_weights())
print(new_model.optimizer)

#loading architecture of the model(priviously saved as json file)

#model_architecture=model_from_json(json_string)

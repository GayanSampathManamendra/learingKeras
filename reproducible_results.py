import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYHONHASHEED']='e'

#setting the seed for numpy generated random numbers

rn.seed(1254)

#setting the seed for tensorflow random numbers

tf.set_random_seed(89)

from keras import backend as k

#force tensorflow to use a single thead

sess=tf.Session(graph=tf.get_default_graph(),config=session_conf)
k.set_session(sess)